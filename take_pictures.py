import sys
import time
import argparse
from picamera import PiCamera

def take_pictures(delay, num_pictures):
    camera = PiCamera()
    camera.resolution = (640, 480)  # Adjust resolution as needed

    for i in range(num_pictures):
        filename = f"/home/rasp/Pictures/picture_{i+1}.jpg"
        camera.capture(filename)
        print(f"Picture {i+1} saved: {filename}")
        time.sleep(delay)

    camera.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take pictures at specific intervals using PiCamera")
    parser.add_argument("delay", type=float, help="Delay between pictures (in seconds)")
    parser.add_argument("num_pictures", type=int, help="Number of pictures to take")

    args = parser.parse_args()
    take_pictures(args.delay, args.num_pictures)
