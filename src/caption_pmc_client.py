#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError

from caption_pkg.srv import *
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse


bridge = CvBridge()
caption = None
img = None


def rgb_callback(image):
	global img,receive_first_image
	img=image


def caption_calculation():
	global img,caption
	try:
		caption=caption_client(img)
	except rospy.ServiceException, e:
		rospy.logwarn("service call failed:%s"%e)


def executeCaption(req):
	global caption
	caption_calculation()
	rospy.sleep(1.)

	return TriggerResponse(
        success=True,
        message=caption.caption_result
    )


if __name__=='__main__':
	rospy.init_node('client_caption_node',anonymous=True)
	image_caption_image_topic = "/"+rospy.get_param("image_caption_image_topic")

	# receice images
	rospy.Subscriber(image_caption_image_topic,Image,rgb_callback)
	while img == None:
		rospy.loginfo('Waiting for images...')
		rospy.sleep(0.5)

	# connect to image_caption service
	rospy.wait_for_service('image_caption')
	caption_client=rospy.ServiceProxy('image_caption',imageCaptioning)

	# create triggerCaption server for main state machine trigger
	triggerCaption_server=rospy.Service('triggerCaption',Trigger,executeCaption)

	rospy.loginfo('Caption client is ready to caption...')
	rospy.spin()
