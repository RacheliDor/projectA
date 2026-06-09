import airsim
print("Attempting reset...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.reset()
print("AirSim vehicle reset complete.")