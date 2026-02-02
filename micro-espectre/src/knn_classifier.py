"""
Micro-ESPectre - KNN Classifier

K-Nearest Neighbors classifier for CSI-based motion classification.
Takes 64*100 subcarrier amplitude values as input.

"""
import math

try:
    from src.detector_interface import IDetector, MotionState
except ImportError:
    from detector_interface import IDetector, MotionState


def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    
    sum_sq = 0.0
    for i in range(len(vec1)):
        diff = vec1[i] - vec2[i]
        sum_sq += diff * diff
    
    return math.sqrt(sum_sq)


class KNNClassifier(IDetector):
    """
    K-Nearest Neighbors classifier for CSI-based classification.
    Supports arbitrary number of classes via multi-class voting.
    
    Input: Array of 64*100 = 6400 subcarrier amplitude values
           (64 subcarriers × 100 time samples)
    
    Args:
        k: Number of nearest neighbors to consider (default: 5)
        window_size: Number of time samples to buffer (default: 100)
        num_subcarriers: Number of CSI subcarriers (default: 64)
        num_classes: Number of classification classes (default: 2 for IDLE/MOTION)
    """
    
    def __init__(self, k=5, window_size=100, num_subcarriers=64, num_classes=2):
        self.k = k
        self.window_size = window_size
        self.num_subcarriers = num_subcarriers
        self.num_classes = num_classes
        
        # Training data storage: list of (feature_vector, label) tuples
        # label: any hashable value (int, str, etc.)
        self.training_data = []
        
        # Sliding window buffer for incoming CSI data
        # Each row is amplitude vector for one time sample
        self.buffer = []
        
        # Current state and predictions
        self.predicted_class = None  # Predicted class label
        self.class_probabilities = {}  # {class_label: probability}
        self.packet_count = 0
        
        # For compatibility with IDetector interface
        self.state = MotionState.IDLE
        self.motion_probability = 0.0
    
    def add_training_sample(self, feature_vector, label):
        """
        Add a labeled training sample.
        
        Args:
            label: Any hashable class label (int, str, tuple, etc.)
        """
        
        # Store as tuple (convert vector to list)
        self.training_data.append((list(feature_vector), label))
    def add_label(self, label):
        """
        Add a new class label to the classifier for the current buffer.
        
        Args:
            label: Any hashable class label (int, str, etc.)
        """
        features=self.calculate_current_features()
        self.add_training_sample(features, label)

    def calculate_current_features(self):
        features=[]
        for sc in range(len(self.buffer[0])):#sc=12
            amplitudes=[self.buffer[i][sc] for i in range(len(self.buffer))]        
            avg_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else 0
            std_dev= math.sqrt(sum((x - avg_amplitude) ** 2 for x in amplitudes) / len(amplitudes)) if amplitudes else 0
            features.append(avg_amplitude)
            features.append(std_dev)
        if len(features) != self.num_subcarriers * 2:
            raise ValueError(f"Feature vector must have {self.num_subcarriers * 2} elements")
        return features

    def load_training_data(self, samples, labels):
        """
        Load multiple training samples at once.
        
        Args:
            samples: List of feature vectors, each with 6400 elements
            labels: List of corresponding class labels (any hashable values)
        """
        if len(samples) != len(labels):
            raise ValueError("Number of samples must match number of labels")
        
        for sample, label in zip(samples, labels):
            self.add_training_sample(sample, label)
    
    def clear_training_data(self):
        """Clear all training data."""
        self.training_data = []
    
    def process_packet(self, csi_data, selected_subcarriers=None):
        amplitudes = []
        for sc in selected_subcarriers:
            idx = (sc - 1) * 2
            real = csi_data[idx]
            imag = csi_data[idx + 1]
            amplitude = (real ** 2 + imag ** 2) ** 0.5
            amplitudes.append(amplitude)
        self.process_amplitudes(amplitudes)

    def process_amplitudes(self, amplitudes):
        """
        Process a single CSI packet by adding it to the sliding window buffer.
        
        Args:
            amplitudes: Array of amplitude values for subcarriers
        """
        self.packet_count += 1
        
        # Ensure we have the expected number of subcarriers
        if len(amplitudes) < self.num_subcarriers:
            # Pad with zeros if needed
            amplitudes.extend([0.0] * (self.num_subcarriers - len(amplitudes)))
        elif len(amplitudes) > self.num_subcarriers:
            # Truncate if too long
            amplitudes = amplitudes[:self.num_subcarriers]
        
        # Add to sliding window buffer
        self.buffer.append(amplitudes)
        
        # Keep buffer at window_size
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
    
    def _flatten_buffer(self):
        """Flatten the buffer into a single feature vector."""
        feature_vector = []
        for time_slice in self.buffer:
            feature_vector.extend(time_slice)
        return feature_vector
    
    def predict(self):
        """
        Classify current buffer using KNN algorithm with multi-class voting.
        
        Returns:
            tuple: (predicted_class, class_probabilities_dict)
                   predicted_class: Most common class label among k-nearest neighbors
                   class_probabilities_dict: {class_label: fraction_of_k_votes}
        """
        # Need full buffer and training data
        if len(self.buffer) < self.window_size:
            return None, {}
        
        if len(self.training_data) == 0:
            return None, {}
        
        # Get feature vector from current buffer
        query_vector = self.calculate_current_features()
        
        # Calculate distances to all training samples
        distances = []
        for train_vector, label in self.training_data:
            dist = euclidean_distance(query_vector, train_vector)
            distances.append((dist, label))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:min(self.k, len(distances))]
        
        # Vote: count votes for each class
        class_votes = {}
        for _, label in k_nearest:
            class_votes[label] = class_votes.get(label, 0) + 1
        
        # Calculate probabilities for each class
        class_probs = {label: votes / len(k_nearest) for label, votes in class_votes.items()}
        
        # Predict the class with most votes
        predicted_class = max(class_votes, key=class_votes.get)
        
        return predicted_class, class_probs
    
    def update_state(self):
        """
        Update classification state based on KNN prediction.
        
        Returns:
            dict: Current metrics including predicted class and probabilities
        """

        if len(self.buffer) >= self.window_size and len(self.training_data) > 0:
            self.predicted_class, self.class_probabilities = self.predict()
        else:
            self.predicted_class = None
            self.class_probabilities = {}
        
        # Update state for IDetector compatibility (IDLE/MOTION if 2 classes)
        if self.predicted_class is not None:
            if self.num_classes == 2:
                # For binary classification, map to IDLE/MOTION
                self.state = self.predicted_class if isinstance(self.predicted_class, int) else MotionState.MOTION
                self.motion_probability = max(self.class_probabilities.values()) if self.class_probabilities else 0.0
            else:
                # For multi-class, set state to predicted class
                self.state = self.predicted_class
                self.motion_probability = max(self.class_probabilities.values()) if self.class_probabilities else 0.0
        else:
            self.state = MotionState.IDLE if self.num_classes == 2 else None
            self.motion_probability = 0.0
        
        return {
            'predicted_class': self.predicted_class,
            'class_probabilities': self.class_probabilities,
            'buffer_fill': len(self.buffer),
            'training_samples': len(self.training_data),
            'packet_count': self.packet_count,
            'state': self.state,  # For compatibility
            'motion_probability': self.motion_probability  # For compatibility
        }
    
    def get_state(self):
        """
        Get current predicted class or state.
        
        Returns:
            The predicted class label, or MotionState.IDLE if no prediction yet
        """
        return self.predicted_class if self.predicted_class is not None else MotionState.IDLE
    
    def get_motion_metric(self):
        """
        Get current confidence metric (probability of predicted class).
        
        Returns:
            float: Confidence of prediction [0.0, 1.0]
        """
        return self.motion_probability
    
    def get_class_probabilities(self):
        """
        Get probability distribution over all classes.
        
        Returns:
            dict: {class_label: probability}
        """
        return self.class_probabilities.copy()
    
    def reset(self):
        """Reset detector state (but keep training data)."""
        self.buffer = []
        self.predicted_class = None
        self.class_probabilities = {}
        self.state = MotionState.IDLE if self.num_classes == 2 else None
        self.motion_probability = 0.0
        self.packet_count = 0


if __name__ == "__main__":
    # Example usage
    sc=12  # Using 12 subcarriers for this example
    window=100
    knn = KNNClassifier(k=3, window_size=window, num_subcarriers=sc, num_classes=3)
    # Add training samples (example data)
    import random
    for _ in range(10):
        for i in range(window):
            knn.process_amplitudes([random.uniform(0, 1) for _ in range(sc)])  # Simulated CSI amplitude
        sample = [random.uniform(0, 1) for _ in range(window*sc)]
        knn.add_label(MotionState.IDLE)
        # knn.add_training_sample(sample, MotionState.IDLE)
    for _ in range(10):
        for i in range(window):
            knn.process_amplitudes([random.uniform(1, 2) for _ in range(sc)])  # Simulated CSI amplitude
        knn.add_label(MotionState.MOTION)
    
    # Simulate processing packets
    for _ in range(100):
        packet = [random.uniform(1, 1.3) for _ in range(sc)]  # Simulated CSI amplitude
        knn.process_amplitudes(packet)
    
    # Update state and get metrics
    metrics = knn.update_state()
    print("Predicted Class:", metrics['predicted_class'])
    print("Class Probabilities:", metrics['class_probabilities'])