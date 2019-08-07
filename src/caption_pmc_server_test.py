#!/usr/bin/env python
import rospy
from caption_pkg.srv import *
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import time
caption_result=""
bridge = CvBridge()

img = None

def rgb_callback(image):
	global img,receive_first_image
	img=image
def caption_calculation():	
	global img
	try:
		
		caption=caption_client(img)
		
		return caption
	except rospy.ServiceException, e:
		rospy.logwarn("service call failed:%s"%e)

if __name__=='__main__':
	rospy.init_node('client_caption_node',anonymous=True)
	#rospy.Subscriber("/c1/camera/rgb/image_raw",Image,rgb_callback)
	rospy.Subscriber("/camera/rgb/image_raw",Image,rgb_callback)
	while img == None:
		time.sleep(0.1)
	
	rospy.wait_for_service('image_caption')
	caption_client=rospy.ServiceProxy('image_caption',imageCaptioning)
	
	rospy.loginfo('Ready to caption')

	caption_result=caption_calculation()
	print (type(caption_result))
	print("The caption is :",caption_result.caption_result)

	