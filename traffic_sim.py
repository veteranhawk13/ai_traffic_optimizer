import random
import time

roads = ["North", "East", "South", "West"]

def simulate_traffic(cycles=5):
    print("🚦 Starting traffic simulation...\n")
    for i in range(cycles):
        active = random.choice(roads)
        print(f"Cycle {i+1}: Green light → {active}")
        time.sleep(1)
    print("\n✅ Simulation complete!")

if __name__ == "__main__":
    simulate_traffic()
