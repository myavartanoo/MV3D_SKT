from loss               import *
from net.blocks         import *



top_view_rpn_name = 'top_view_rpn'



def VGG_top_net(input, anchors, inds_inside, num_bases):
    print('build_VGG')


    stride=1.

    with tf.variable_scope('top-features_top-1') as scope:
        features_top = conv2d_bn_relu(input, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        features_top = conv2d_bn_relu(features_top, num_kernels=32, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        features_top = maxpool(features_top, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-features_top-2') as scope:
        features_top = conv2d_bn_relu(features_top, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        features_top = conv2d_bn_relu(features_top, num_kernels=64, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        features_top = maxpool(features_top, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-features_top-3') as scope:
        features_top = conv2d_bn_relu(features_top, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        features_top = conv2d_bn_relu(features_top, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        features_top = conv2d_bn_relu(features_top, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')
        features_top = maxpool(features_top, kernel_size=(2,2), stride=[1,2,2,1], padding='SAME', name='4' )
        stride *=2

    with tf.variable_scope('top-features_top-4') as scope:
        features_top = conv2d_bn_relu(features_top, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='1')
        features_top = conv2d_bn_relu(features_top, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        features_top = conv2d_bn_relu(features_top, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='3')


    with tf.variable_scope('top') as scope:
        up      = conv2d_bn_relu(features_top, num_kernels=128, kernel_size=(3,3), stride=[1,1,1,1], padding='SAME', name='2')
        scores_top  = conv2d(up, num_kernels=2*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='score')
        probs_top   = tf.nn.softmax( tf.reshape(scores_top,[-1,2]), name='prob')
        deltas_top  = conv2d(up, num_kernels=4*num_bases, kernel_size=(1,1), stride=[1,1,1,1], padding='SAME', name='delta')

    with tf.variable_scope('top-nms') as scope:    #non-max
        batch_size, img_height, img_width, img_channel = input.get_shape().as_list()
        img_scale = 1
        rois_top, roi_scores_top = tf_rpn_nms( probs_top, deltas_top, anchors, inds_inside,
                                       stride, img_width, img_height, img_scale,
                                       nms_thresh=0.7, min_size=stride, nms_pre_topn=500, nms_post_topn=100,
                                       name ='nms')


    print ('top: scale=%f, stride=%d'%(1./stride, stride))
    return features_top, scores_top, probs_top, deltas_top, rois_top, roi_scores_top, stride
