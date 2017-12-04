import tensorflow as tf


def modified_smooth_l1( deltas, targets, sigma=3.0, param=0.5):
    '''
        ResultLoss = outside_weights * SmoothL1(inside_weights * (box_pred - box_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    '''
    sigma2 = sigma **2
    dif  =  tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(dif), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(dif, dif) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(dif) - param / sigma2
    smooth_l1 = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)

    return smooth_l1


#------------------------------------------------------------------------------

def rpn_loss(scores, deltas, inds, pos_inds, rpn_labels, rpn_targets):


    scores1      = tf.reshape(scores,[-1,2])
    rpn_scores   = tf.gather(scores1,inds)  # remove ignore label
    rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_scores, labels=rpn_labels))

    deltas1       = tf.reshape(deltas,[-1,4])
    rpn_deltas    = tf.gather(deltas1, pos_inds)  # remove ignore label

    with tf.variable_scope('modified_smooth_l1'):
        rpn_smooth_l1 = modified_smooth_l1(rpn_deltas, rpn_targets, sigma=3.0, param=0.0)

    rpn_reg_loss  = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, axis=1))
    return rpn_cls_loss, rpn_reg_loss


