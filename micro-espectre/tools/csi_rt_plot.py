import threading
import queue
import sys
import select
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from csi_utils import CSIPacket, CSIPacket, CSIPacket, CSIReceiver
from src.knn_classifier import KNNClassifier

# Create custom colormap: 0=black, 100=yellow
black_yellow_cmap = LinearSegmentedColormap.from_list(
    'black_yellow',
    ['black', 'yellow']
)

class SpectrogramRTPlot:
    """
    Real-time spectrogram-style plot with shape (height=64, width=50).
    X axis (width=50) represents time (most recent column on the right).
    """

    def __init__(self, width=50, height=64, cmap='black_yellow', vmin=0, vmax=100):
        self.width = width
        self.height = height
        self.buffer = np.zeros((height, width), dtype=float)
        self.cmap = black_yellow_cmap if cmap == 'black_yellow' else cmap
        self.update_queue = queue.Queue()  # Thread-safe queue for updates

        self.selected_subcarriers =[11, 12, 13, 17, 44, 45, 46, 48, 49, 50, 51, 52]# [28, 38, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50]

        self.knnClassifier = KNNClassifier(k=3, window_size=100, num_subcarriers=len(self.selected_subcarriers), num_classes=3)
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.im = self.ax.imshow(
            self.buffer,
            aspect='auto',
            origin='lower',
            interpolation='nearest',
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, self.width, 0, self.height],
        )

        self.ax.set_xlabel('time (frames)')
        self.ax.set_ylabel('frequency bin')
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.fig.colorbar(self.im, ax=self.ax, label='amplitude')
        plt.tight_layout()
        plt.ion()
        self.fig.show()
        self.fig.canvas.draw()

    def update_column(self, column):
        """
        Push a new column (1D array of length `height`) into the spectrogram.
        The column becomes the newest (right-most) time slice.
        Safe to call from worker threads.
        """
        col = np.asarray(column, dtype=float)
        if col.size != self.height:
            raise ValueError(f"column length must be {self.height}, got {col.size}")
        # Queue the update to be processed in the main thread
        self.update_queue.put({'type': 'column', 'data': col})

    def update_matrix(self, matrix):
        """
        Replace the whole matrix with a (height x width) array.
        Safe to call from worker threads.
        """
        mat = np.asarray(matrix, dtype=float)
        if mat.shape != (self.height, self.width):
            raise ValueError(f"matrix must have shape {(self.height, self.width)}, got {mat.shape}")
        # Queue the update to be processed in the main thread
        self.update_queue.put({'type': 'matrix', 'data': mat})

    def _refresh(self):
        # Process any pending updates from worker threads
        if self.update_queue.empty():
            return
        #average the values for each column to create the new buffer entry
        sumbuf=None
        count=0
        while not self.update_queue.empty():
            try:
                update_data = self.update_queue.get_nowait()
                if update_data['type'] == 'column':
                    col = np.asarray(update_data['data'], dtype=float)
                    if col.size == self.height:
                        if sumbuf is None:
                            sumbuf=col
                        else:
                            sumbuf=sumbuf+col
                        count += 1
            except queue.Empty:
                break

        if sumbuf is not None and count>0:
            avgcol=sumbuf/count
            self.buffer = np.roll(self.buffer, -1, axis=1)
            self.buffer[:, -1] = avgcol
        
        # while not self.update_queue.empty():
        #     try:
        #         update_data = self.update_queue.get_nowait()
        #         if update_data['type'] == 'column':
        #             col = np.asarray(update_data['data'], dtype=float)
        #             if col.size == self.height:
        #                 self.buffer = np.roll(self.buffer, -1, axis=1)
        #                 self.buffer[:, -1] = col
        #         elif update_data['type'] == 'matrix':
        #             mat = np.asarray(update_data['data'], dtype=float)
        #             if mat.shape == (self.height, self.width):
        #                 self.buffer = mat.copy()
        #     except queue.Empty:
        #         break
        
        self.im.set_data(self.buffer)
        # keep colormap autoscaling stable if desired; otherwise comment out
        # self.im.set_clim(vmin=self.buffer.min(), vmax=self.buffer.max())
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def close(self):
        plt.close(self.fig)

    def start_receiver(self,port):
        """
        Start to receive CSI data on the given UDP port.
        """
        self.receiver = CSIReceiver(port=port, buffer_size=2000)
        self.receiver.add_callback(self._process_csi_packet)

        threading.Thread(target=self._receive_loop, daemon=True).start()

    def _receive_loop(self):
        self.receiver.run()
    def _process_csi_packet(self, csi_packet:CSIPacket):
        # add to spectrogram
        amplitude = csi_packet.amplitudes
        selected_amplitudes = amplitude[self.selected_subcarriers]
        self.knnClassifier.process_amplitudes(selected_amplitudes)
        # amps_normalized = 50+(amplitude - np.average(selected_amplitudes))#) / (np.max(selected_amplitudes) - np.min(selected_amplitudes) + 1e-6) * 100
        # self.update_column(amps_normalized)
        self.update_column(amplitude)

if __name__ == "__main__":
    import time
    plot = SpectrogramRTPlot()

    plot.start_receiver(port=5001)
    print("Starting real-time CSI spectrogram plot. \n" \
    "In another terminal, run the CSI streamer to send data to UDP port 5001, for example:\n" \
        "./me stream --ip 100.64.10.237 ")
    print("Type a label and press Enter to add a KNN training sample from the current buffer.")

    def poll_user_line():
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line == "":
                return None
            return line.rstrip("\n")
        return None

    try:
        while True:
            user_line = poll_user_line()
            if user_line is not None and user_line.strip():
                plot.knnClassifier.add_label(user_line.strip())
                print(f"Added KNN label: {user_line.strip()}")
            plot._refresh()
            metrics=plot.knnClassifier.update_state()
            print("Predicted Class:", metrics['predicted_class'])
            print("Class Probabilities:", metrics['class_probabilities'])
            print("packets Processed:", plot.knnClassifier.packet_count)
            plot.knnClassifier.packet_count=0

            # if plot.knnClassifier.buffer: 
            #     print("features:", plot.knnClassifier.calculate_current_features())
            time.sleep(0.3)
            

        #test data
        for i in range(100):
            new_col = np.random.rand(plot.height) * 100  # random data for testing
            plot.update_column(new_col)
            plot._refresh()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        plot.close()