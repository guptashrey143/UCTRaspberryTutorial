from gpiozero import LED
import time

ledpin = LED(17)



for i in range(5):
    print("LED turning on.")
    ledpin.on()
    time.sleep(2)
    print("LED turning off.")
    ledpin.off()
    time.sleep(1)
