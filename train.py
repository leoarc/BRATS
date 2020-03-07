#27097, 27342
from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys,csv
import subprocess
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=1, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=480, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=31, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=True, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=True, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=0.1, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=10, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="MobileUNet-Skip", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')

parser.add_argument('--checkpoint_foldername', type=str, default="0", help='create folder checkpoint_somenumber')

args = parser.parse_args()

if not os.path.isdir("%s_%s"%("checkpoint",args.checkpoint_foldername)):
    os.makedirs("%s_%s"%("checkpoint",args.checkpoint_foldername))

#pwd = os.getcwd()
#a= pwd + "/checkpoint_" + args.checkpoint_foldername
#completed_epoch = len(next(os.walk(a))[1])
csvfile_name= "%s_%s.csv"%("checkpoint",args.checkpoint_foldername)

if not args.continue_training:
    csvData = [['Epoch No.', 'avg scores per epoch', 'avg loss per epoch', 'avg iou per epoch', 'recall benign', 'precision benign', 'recall malignant', 'precision malignant','F Score_mal','F Score_ben','F Score']]
    #row= [epoch, avg_scores_per_epoch, avg_loss_per_epoch, avg_iou_per_epoch, recall_benign, prec_benign, recall_malignant, prec_malignant]
    csvfile_name= "%s_%s.csv"%("checkpoint",args.checkpoint_foldername)
    
    with open(csvfile_name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()
    completed_epoch=0
else:
    with open(csvfile_name, 'r') as csvFile:
        x= sum(1 for row in csvFile)
        completed_epoch = int(x)-1
def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image

########################################################################################################################################
"""def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    
    #This function prints and plots the confusion matrix.
    #Normalization can be applied by setting `normalize=True`.
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax"""



############################################################################################################################################

# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

##NEW LOSS FUNCTION
weight=tf.constant([5.0,5.0,3.0,1.0])
loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=network, targets=net_output,pos_weight=weight))

opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.8).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

#avg_loss_per_epoch = []
#avg_scores_per_epoch = []
# avg_iou_per_epoch = []

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoint_" + args.checkpoint_foldername + "/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)



print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")



# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)

# Do the training here
for epoch in range(args.epoch_start_i+completed_epoch, args.num_epochs):

    current_losses = []

    cnt=0

    # Equivalent to shuffling
    id_list = np.random.permutation(len(train_input_names))

    num_iters = int(np.floor(len(id_list) / args.batch_size))
    #num_iters=800
    st = time.time()
    epoch_st=time.time()
    for i in range(num_iters):
        # st=time.time()

        input_image_batch = []
        output_image_batch = []

        # Collect a batch of images
        for j in range(args.batch_size):
            index = i*args.batch_size + j
            id = id_list[index]
            input_image = utils.load_image(train_input_names[id])
            output_image = utils.load_image(train_output_names[id])

            with tf.device('/cpu:0'):
                input_image, output_image = data_augmentation(input_image, output_image)


                # Prep the data. Make sure the labels are in one-hot format
                input_image = np.float32(input_image) / 255.0
                output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

                input_image_batch.append(np.expand_dims(input_image, axis=0))
                output_image_batch.append(np.expand_dims(output_image, axis=0))

        if args.batch_size == 1:
            input_image_batch = input_image_batch[0]
            output_image_batch = output_image_batch[0]
        else:
            input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
            output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

        # Do the training
        _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
        current_losses.append(current)
        cnt = cnt + args.batch_size
        if cnt % 20 == 0:
            string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
            utils.LOG(string_print)
            st = time.time()

    mean_loss = np.mean(current_losses)
    avg_loss_per_epoch=np.round(mean_loss,3)

    # Create directories if needed
    if not os.path.isdir("%s_%s/%04d"%("checkpoint",args.checkpoint_foldername,epoch)):
        os.makedirs("%s_%s/%04d"%("checkpoint",args.checkpoint_foldername,epoch))

    # Save latest checkpoint to same file name
    print("Saving latest checkpoint")
    saver.save(sess,model_checkpoint_name)

    if val_indices != 0 and epoch % args.checkpoint_step == 0:
        print("Saving checkpoint for this epoch")
        saver.save(sess,"%s_%s/%04d/model.ckpt"%("checkpoint",args.checkpoint_foldername,epoch))


    if epoch % args.validation_step == 0:
        print("Performing validation")
        target=open("%s_%s/%04d/val_scores.csv"%("checkpoint",args.checkpoint_foldername,epoch),'w')
        target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))


        scores_list = []
        class_scores_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        iou_list = []

        Sum=0
        # Do the validation on a small set of validation images
        for ind in val_indices:

            input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:1440,:1920]),axis=0)/255.0
            #print("--------input_image size in validation   ",input_image.shape)
            gt = utils.load_image(val_output_names[ind])[:1440, :1920]
            #print("---------output_image size in validation   ",gt.shape)
            #print('\n----------gt is {}'.format(gt))
            gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))
            #print('\n------------GT IS ',gt)
            # st = time.time()

            output_image = sess.run(network,feed_dict={net_input:input_image})


            output_image = np.array(output_image[0,:,:,:])
            output_image = helpers.reverse_one_hot(output_image)
            out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
           
    #############################################################################################################################
            y_test=gt.reshape(1,gt.shape[0]*gt.shape[1])
            y_test=np.squeeze(y_test)
            
            y_test=np.append(y_test,[0,1,2,3])
            
            y_pred=output_image.reshape(1,gt.shape[0]*gt.shape[1])
            y_pred= np.squeeze(y_pred)
            
            y_pred=np.append(y_pred,[0,1,2,3])
            #np.set_printoptions(precision=2)
            Sum=Sum+confusion_matrix(y_test,y_pred)
            #print(confusion_matrix(y_test,y_pred))

# Plot non-normalized confusion matrix
            #plot_confusion_matrix(y_test, y_pred, classes=class_names,
            #title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
            #plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,title='Normalized confusion matrix')

            #plt.show()
#########################################################################################################################################            
            accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

            file_name = utils.filepath_to_name(val_input_names[ind])
            target.write("%s, %0.3f, %0.3f, %0.3f, %0.3f, %0.3f"%(file_name, accuracy, prec, rec, f1, iou))
            for item in class_accuracies:
                target.write(", %0.3f"%(item))
            target.write("\n")

            scores_list.append(accuracy)
            class_scores_list.append(class_accuracies)
            precision_list.append(prec)
            recall_list.append(rec)
            f1_list.append(f1)
            iou_list.append(iou)

            gt = helpers.colour_code_segmentation(gt, label_values)

            file_name = os.path.basename(val_input_names[ind])
            file_name = os.path.splitext(file_name)[0]
            cv2.imwrite("%s_%s/%04d/%s_pred.png"%("checkpoint",args.checkpoint_foldername,epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
            cv2.imwrite("%s_%s/%04d/%s_gt.png"%("checkpoint",args.checkpoint_foldername,epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


        target.close()

        avg_score = np.mean(scores_list)
        class_avg_scores = np.mean(class_scores_list, axis=0)
        avg_scores_per_epoch=np.round(avg_score,3)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_iou = np.mean(iou_list)
        avg_iou_per_epoch=np.round(avg_iou,3)

        print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
        print("Average per class validation accuracies for epoch # %04d:"% (epoch))
        for index, item in enumerate(class_avg_scores):
            print("%s = %f" % (class_names_list[index], item))
        print("Validation precision = ", avg_precision)
        print("Validation recall = ", avg_recall)
        print("Validation F1 score = ", avg_f1)
        print("Validation IoU score = ", avg_iou)
    
    csvData = ['True, predicted-->.', 'benign', 'malignant', 'isolated', 'white','recall']
    recall_benign=Sum[0][0]/(sum(Sum[0,:]))
    prec_benign=Sum[0][0]/(sum(Sum[:,0]))
    recall_malignant=Sum[1][1]/(sum(Sum[1,:]))
    prec_malignant=Sum[1][1]/(sum(Sum[:,1]))
    recall_isolated=Sum[2][2]/(sum(Sum[2,:]))
    prec_isolated=Sum[2][2]/(sum(Sum[:,2]))
    recall_white=Sum[3][3]/(sum(Sum[3,:]))
    prec_white=Sum[3][3]/(sum(Sum[:,3]))

    

    recall=[recall_benign, recall_malignant, recall_isolated, recall_white]
    prec=['precision', prec_benign, prec_malignant, prec_isolated , prec_white]
    confusion_mat_file="%s_%s/%04d.csv"%("checkpoint",args.checkpoint_foldername,epoch)
    with open(confusion_mat_file,'w') as csvFile:
        writer=csv.writer(csvFile)
        writer.writerow(csvData)
        j=1
        for row in Sum:
            cs=[]
            cs.append(csvData[int(j)])
            for i in row:
                cs.append(i)
            cs.append(recall[j-1])
            writer.writerow(cs)
            j=1+j
        writer.writerow(prec)

    print('\n--------confusion matrix after epoch {} is \n {}'.format(epoch,Sum))
    print('\n-------sumof elements is ',np.sum(Sum))
    epoch_time=time.time()-epoch_st
    remain_time=epoch_time*(args.num_epochs-1-epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if s!=0:
        train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
    else:
        train_time="Remaining training time : Training completed.\n"
    utils.LOG(train_time)
    scores_list = []
    recall_benign=np.round(Sum[0][0]/(sum(Sum[0,:])),3)
    prec_benign=np.round(Sum[0][0]/(sum(Sum[:,0])),3)
    recall_malignant=np.round(Sum[1][1]/(sum(Sum[1,:])),3)
    prec_malignant=np.round(Sum[1][1]/(sum(Sum[:,1])),3)
    F_score_mal = np.round(2*prec_malignant*recall_malignant/(prec_malignant+recall_malignant),3)
    F_score_ben = np.round(2*prec_benign*recall_benign/(prec_benign+recall_benign),3)
    F_score = np.round((F_score_mal + F_score_ben)/2,3)
    #Dice_score= 2*Sum[1][1]/(sum(Sum[1,:])+sum(Sum[:,1]))
    #appending data in csvfile
    row= [epoch, avg_scores_per_epoch, avg_loss_per_epoch, avg_iou_per_epoch, recall_benign, prec_benign, recall_malignant, prec_malignant, F_score_mal , F_score_ben , F_score]
    csvfile_name= "%s_%s.csv"%("checkpoint",args.checkpoint_foldername)
    with open(csvfile_name, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()







    """fig1, ax1 = plt.subplots(figsize=(11, 8))

    ax1.plot(range(epoch+1), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")


    plt.savefig('accuracy_vs_epochs_'+args.checkpoint_foldername+'.png')

    plt.clf()

    fig2, ax2 = plt.subplots(figsize=(11, 8))

    ax2.plot(range(epoch+1), avg_loss_per_epoch)
    ax2.set_title("Average loss vs epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs_'+args.checkpoint_foldername+'.png')

    plt.clf()

    fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch+1), avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")

    plt.savefig('iou_vs_epochs_'+args.checkpoint_foldername+'.png')"""
