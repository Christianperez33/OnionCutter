import serial
import serial.tools.list_ports
import time
import cv2

def find_arduino():
	for pinfo in serial.tools.list_ports.comports():
		if "Arduino Uno" in list(pinfo):
			return serial.Serial(pinfo.device,115200)
	raise IOError("Not found arduino- plug it.")

arduino = find_arduino();
vel=100
key = cv2.waitKey(1) & 0xFF

while key != ord('q'):
	var=vel+100
	arduino.write(str(var).encode())
	if vel>=255:
		vel=100
	else:
		vel+=50
	time.sleep(1)


	arduino = find_arduino();
	key = cv2.waitKey(1) & 0xFF

arduino.close()