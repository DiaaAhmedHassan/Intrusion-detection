import torch
import torch.nn as nn
from scapy.all import sniff
from datetime import datetime

class IntrusionModel(nn.Module):
    def __init__(self):
        super(IntrusionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = IntrusionModel()
model.load_state_dict(torch.load("./save/intrusion_model_weights.pth", map_location=device))
model.eval()

def classify_packet(packet):
    features = extract_features(packet)
    if features:
        x = torch.tensor([features], dtype=torch.float32)
        output = model(x)
        label = "Attack" if output.item() > 0.5 else "Normal"
        log_entry = f"[{datetime.now()}] {packet.summary()} => {label}\n"
        print(log_entry.strip())

        log_file = f'./logs/packet_log_{datetime.today()}.txt'
        with open(log_file, "a") as log_file:
            log_file.write(log_entry)

def extract_features(packet):
    try:
        length = len(packet)
        proto = 1 if packet.haslayer("TCP") else 0
        login_attempts = 3
        session_duration = 500.0
        encryption = 0
        reputation_score = 0.5
        failed_logins = 1
        unusual_time = 0
        source_bytes = 1000
        dest_bytes = 800
        num_root = 0
        
        # browser info
        browser = 'unknown'
        # if packet.haslayer('Raw'):
        #     payload = packet['Raw'].load
        #     if b"User-Agent" in payload:
        #         browser = payload.split(b"User-Agent:")[1].split(b"\r\n")[0]

        return [length, proto, login_attempts, session_duration,
                encryption, reputation_score, failed_logins,
                browser, unusual_time, source_bytes, dest_bytes, num_root, 0]
    except:
        return None

print("Sniffing started... Press Ctrl+C to stop.")

try: 
    sniff(prn=classify_packet, store=0)
except Exception as e:
    print(f"Error: {e}")