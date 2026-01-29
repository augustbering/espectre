"""
Micro-ESPectre - Ping (ICMP) Traffic Generator

Generates ICMP ping requests to a specified server to force bidirectional
traffic (useful to trigger CSI packet replies). Mirrors the API of
`traffic_generator.py` but uses ping requests instead of DNS queries.

This implementation uses the system `ping` command when available. It is
designed to run on a host Linux environment for testing; root privileges
may be required for raw ICMP sockets if an alternative implementation
is used.
"""
import time
import _thread
import socket
import network

TRAFFIC_RATE_MIN = 0
TRAFFIC_RATE_MAX = 1000
METRICS_INTERVAL = 500


class TrafficGeneratorEcho:
    """Ping-based traffic generator (ICMP echo requests)

    Methods mirror the original `TrafficGenerator` where practical.
    """

    def __init__(self):
        self.running = False
        self.rate_pps = 0
        self.packet_count = 0
        self.error_count = 0
        self.server = None
        self.start_time = 0
        self.avg_loop_time_ms = 0
        self.actual_pps = 0


    def _get_gateway_ip(self):
        try:
            wlan = network.WLAN(network.STA_IF)
            if not wlan.isconnected():
                return None
            ip_info = wlan.ifconfig()
            if len(ip_info) >= 3:
                return ip_info[2]
            return None
        except Exception:
            return None

    def _resolve_server(self):
        # If server provided, resolve it; otherwise use gateway IP
        if self.server:
            ## assume it's an IP address for now
            return self.server
            try:
                return socket.gethostbyname(self.server)
            except Exception:
                return None

        return self._get_gateway_ip()

    def _now_us(self):
        return time.ticks_us()

    def _icmp_checksum(self, data: bytes) -> int:
        s = 0
        # Sum 16-bit words
        for i in range(0, len(data) - 1, 2):
            s += (data[i] << 8) + data[i + 1]
        if len(data) % 2:
            s += data[-1] << 8
        s = (s >> 16) + (s & 0xFFFF)
        s += s >> 16
        return (~s) & 0xFFFF

    def _build_icmp_packet(self, id_val, seq, payload=b"\x00"):
        header = bytearray(8)
        header[0] = 8  # Type: 8 = Echo Request
        header[1] = 0  # Code
        header[2] = 0  # checksum placeholder
        header[3] = 0
        header[4] = (id_val >> 8) & 0xFF
        header[5] = id_val & 0xFF
        header[6] = (seq >> 8) & 0xFF
        header[7] = seq & 0xFF
        packet = bytes(header) + payload
        chksum = self._icmp_checksum(packet)
        packet = bytearray(packet)
        packet[2] = (chksum >> 8) & 0xFF
        packet[3] = chksum & 0xFF
        return bytes(packet)

    def _icmp_task(self):
        dest_ip = self._resolve_server()
        print("Ping Traffic Generator: Pinging", dest_ip)
        if not dest_ip:
            self.error_count += 1
            self.running = False
            return

        # Try to create a raw ICMP socket
        try:
            proto = getattr(socket, 'IPPROTO_ICMP', 1)
            sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, proto)
            try:
                sock.setblocking(False)
            except Exception:
                pass
        except Exception as e:
            self.error_count += 1
            self.running = False
            return

        self.sock = sock

        loop_time_sum_us = 0
        window_start = self._now_us()
        window_count = 0

        if self.rate_pps <= 0:
            self.running = False
            sock.close()
            self.sock = None
            return

        interval_us = 1000000 // self.rate_pps
        remainder_us = 1000000 % self.rate_pps
        accumulator = 0

        next_send = self._now_us()
        seq = 0
        id_val = (int(self.start_time) & 0xFFFF) if self.start_time else (self._now_us() & 0xFFFF)

        payload = b"\x00"

        while self.running:
            try:
                loop_start = self._now_us()

                seq = (seq + 1) & 0xFFFF
                pkt = self._build_icmp_packet(id_val, seq, payload)
                try:
                    sock.sendto(pkt, (dest_ip, 0))
                    self.packet_count += 1
                    window_count += 1
                except OSError:
                    self.error_count += 1

                accumulator += remainder_us
                extra = accumulator // self.rate_pps
                accumulator %= self.rate_pps
                next_send += interval_us + extra

                loop_time_us = time.ticks_diff(self._now_us(), loop_start)
                loop_time_sum_us += loop_time_us

                if window_count >= METRICS_INTERVAL:
                    self.avg_loop_time_ms = (loop_time_sum_us / METRICS_INTERVAL) / 1000
                    loop_time_sum_us = 0
                    elapsed = time.ticks_diff(self._now_us(), window_start)
                    if elapsed > 0:
                        self.actual_pps = (window_count * 1000000) / elapsed
                    window_start = self._now_us()
                    window_count = 0

                now = self._now_us()
                sleep_us = time.ticks_diff(next_send, now)

                if sleep_us > 1000:
                    time.sleep_ms(sleep_us // 1000)
                elif sleep_us < -100000:
                    next_send = self._now_us()
                else:
                    time.sleep_us(100)

            except Exception:
                self.error_count += 1
                time.sleep_ms(1)

        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def start(self, rate_pps, max_retries=3, retry_delay=2):
        """Start ping-based generator.

        Args:
            rate_pps: packets per second (0-1000)
        """
        if self.running:
            return False

        if rate_pps < TRAFFIC_RATE_MIN or rate_pps > TRAFFIC_RATE_MAX:
            return False


        # Ensure server resolves
        resolved = self._resolve_server()
        # for attempt in range(1, max_retries + 1):
        #     resolved = self._resolve_server()
        #     if resolved:
        #         break
        #     time.sleep(retry_delay)
        print("Ping Traffic Generator: Resolved server to", resolved)

        if not resolved:
            raise Exception("No server to ping to!")

        print("Ping Traffic Generator: Starting with rate", rate_pps, "pps")

        self.packet_count = 0
        self.error_count = 0
        self.rate_pps = rate_pps
        self.start_time = int(time.time() * 1000)
        self.running = True

        try:
            _thread.start_new_thread(self._icmp_task, ())
            return True
        except Exception:
            self.running = False
            print("Ping Traffic Generator: Failed to start")
            return False

    def stop(self):
        if not self.running:
            return
        self.running = False
        time.sleep(0.5)
        self.rate_pps = 0

    def is_running(self):
        return self.running

    def get_packet_count(self):
        return self.packet_count

    def get_rate(self):
        return self.rate_pps

    def get_actual_pps(self):
        return round(self.actual_pps, 1)

    def get_error_count(self):
        return self.error_count

    def get_avg_loop_time_ms(self):
        return round(self.avg_loop_time_ms, 2)

# tg = TrafficGeneratorEcho()
# tg.server="100.64.10.237"
# tg.start(rate_pps=10)