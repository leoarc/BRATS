from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np
import csv

from utils import utils, helpers
from builders import model_builder
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default="checkpoint_", required=False, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=320, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=320, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default="MobileUNet-Skip", required=False, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
parser.add_argument('--checkpoint_foldername', type=str, default="0", help='create folder checkpoint_somenumber')
parser.add_argument('--output_foldername', type=str, default="00", help='create folder output folder')
parser.add_argument('--epoch_number',type=str,default="0001")

args = parser.parse_args()
#model_checkpoint_name = "checkpoint_" + args.checkpoint_foldername + "/latest_model_" + "MobileUNet" + "_" + args.dataset + ".ckpt"
model_checkpoint_name =  "checkpoint_" + args.checkpoint_foldername + "/" + args.epoch_number + "/model.ckpt"
print(model_checkpoint_name)
# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

# Create directories if needed
if not os.path.isdir("%s_%s"%("Test",args.output_foldername)):
        os.makedirs("%s_%s"%("Test",args.output_foldername))


st1 = time.time()
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1,len(test_input_names)))
    sys.stdout.flush()
    inp_img=np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width])
    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    
    
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
    file_name = utils.filepath_to_name(test_input_names[ind])
    cv2.imwrite("%s_%s/%s_org.png"%("Test",args.output_foldername, file_name),cv2.cvtColor(np.uint8(inp_img), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s_%s/%s_pred.png"%("Test",args.output_foldername, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))

