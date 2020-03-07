#!/usr/bin/env python2
import rospy
from cv_bridge import CvBridge, CvBridgeError

from caption_pkg.srv import *
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import String

caption_client = None
bridge = CvBridge()
caption = None
img = None
pub = None


def rgb_callback(image):
	global img,receive_first_image
	img=image


def caption_calculation():
	global img,caption, caption_client
	try:
		caption=caption_client(img)
	except rospy.ServiceException, e:
		rospy.logwarn("service call failed:%s"%e)


def executeCaption(req):
	global caption,pub
	caption_calculation()
	pub.publish(String(str(caption.caption_result)))
	rospy.logwarn("captioning:: %s",str(caption.caption_result))
	rospy.sleep(3.)

	return TriggerResponse(
        success=True,
        message=caption.caption_result
    )


def main():
	global pub, caption_client
	rospy.init_node('client_caption_node',anonymous=True)
	image_caption_image_topic = "/"+rospy.get_param("image_caption_image_topic")

	# receice images
	rospy.Subscriber(image_caption_image_topic,Image,rgb_callback)
	while img == None:
		rospy.loginfo('Waiting for images...')
		rospy.sleep(0.5)
		if rospy.is_shutdown():
			exit(-1)

	# connect to image_caption service
	rospy.wait_for_service('image_caption')
	caption_client=rospy.ServiceProxy('image_caption',imageCaptioning)

	# create triggerCaption server for main state machine trigger
	triggerCaption_server=rospy.Service('triggerCaption',Trigger,executeCaption)

	# vocal part
	pub = rospy.Publisher('response', String, queue_size=10)

	rospy.logwarn('Caption client is ready to caption...')
	rospy.spin()


if __name__=='__main__':
    main()
