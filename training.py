import tensorflow as tf

import network
import io
import pickle
import subprocess
import sys

from net.rpn_nms_op         import *
from utils                  import *
from network                import *
from config_rpn             import cfg
from keras                  import backend as K
from time                   import localtime, strftime

def load_net(shape_top, num_bases_top):

    anchors_top     = tf.placeholder(shape=[None, 4], dtype=tf.int32, name='anchors')
    inside_inds_top = tf.placeholder(shape=[None], dtype=tf.int32, name='inside_inds')
    view_top        = tf.placeholder(shape=[None, *shape_top], dtype=tf.float32, name='top')
    rois_top        = tf.placeholder(shape=[None, 5], dtype=tf.float32, name='rois_top')
    with tf.variable_scope(top_view_rpn_name):
        # top feature
        features_top, scores_top, probs_top, deltas_top, proposals_top, proposal_scores_top, feature_stride_top = \
            VGG_top_net(view_top, anchors_top, inside_inds_top, num_bases_top)

        with tf.variable_scope('loss'):
            # RRN
            inds_top                    = tf.placeholder(shape=[None], dtype=tf.int32, name='top_ind')
            pos_inds_top                = tf.placeholder(shape=[None], dtype=tf.int32, name='top_pos_ind')
            labels_top                  = tf.placeholder(shape=[None], dtype=tf.int32, name='top_label')
            targets_top                 = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='top_target')
            cls_loss_top, reg_loss_top  = rpn_loss(scores_top, deltas_top, inds_top, pos_inds_top, labels_top, targets_top)
    MyNetwork                   = type('MyNetwork', (object,), {})
    net                         = MyNetwork()
    net.anchors_top             = anchors_top
    net.inside_inds_top         = inside_inds_top
    net.view_top                = view_top
    net.rois_top                = rois_top
    net.cls_loss_top            = cls_loss_top
    net.reg_loss_top            = reg_loss_top
    net.features_top            = features_top
    net.scores_top              = scores_top
    net.probs_top               = probs_top
    net.deltas_top              = deltas_top
    net.proposals_top           = proposals_top
    net.proposal_scores_top     = proposal_scores_top
    net.inds_top                = inds_top
    net.pos_inds_top            = pos_inds_top
    net.labels_top              = labels_top
    net.targets_top             = targets_top
    net.feature_stride_top      = feature_stride_top

    return net




class Logger(object):
    def __init__(self,file=None, mode=None):
        self.terminal = sys.stdout
        self.file     = None
        if file is not None: self.open(file,mode)


    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if message =='\r': is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


class Net(object):

    def __init__(self, prefix, scope_name, checkpoint_dir=None):
        self.name                   =scope_name
        self.prefix                 = prefix
        self.checkpoint_dir         =checkpoint_dir
        self.subnet_checkpoint_dir  = os.path.join(checkpoint_dir, scope_name)
        self.subnet_checkpoint_name = scope_name
        os.makedirs(self.subnet_checkpoint_dir, exist_ok=True)
        self.variables              = self.get_variables([prefix+'/'+scope_name])
        self.saver                  = tf.train.Saver(self.variables)


    def save_weights(self, sess=None):
        path = os.path.join(self.subnet_checkpoint_dir, self.subnet_checkpoint_name)
        print('\nSave weigths : %s' % path)
        self.saver.save(sess, path)

    def clean_weights(self):
        command = 'rm -rf %s' % (os.path.join(self.subnet_checkpoint_dir))
        subprocess.call(command, shell=True)
        print('\nClean weights: %s' % command)
        os.makedirs(self.subnet_checkpoint_dir ,exist_ok=True)


    def load_weights(self, sess=None):
        path = os.path.join(self.subnet_checkpoint_dir, self.subnet_checkpoint_name)
        if tf.train.checkpoint_exists(path) ==False:
            print('\nCan not found :\n"%s",\nuse default weights instead it\n' % (path))
            path = path.replace(os.path.basename(self.checkpoint_dir),'default')
        assert tf.train.checkpoint_exists(path) == True
        self.saver.restore(sess, path)


    def get_variables(self, scope_names):
        variables=[]
        for scope in scope_names:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
            assert len(variables) != 0
            variables += variables
        return variables


class MV3D(object):

    def __init__(self, top_shape, debug_mode=False, dist_name=None, weigths_dir=None):

        # anchors
        self.top_stride=None
        self.num_class = 6  # incude background

        ratios=np.array([0.5,1,2], dtype=np.float32)
        scales=np.array([1,2,3],   dtype=np.float32)


        self.bases = make_bases(
            base_size = 16,
            ratios=ratios,  #aspet ratio
            scales=scales
        )

        # output dir, etc
        if not os.path.isdir(cfg.CHECKPOINT_DIR):
            os.makedirs(cfg.CHECKPOINT_DIR)
        self.log_msg    = Logger(cfg.LOG_DIR + '/log.txt', mode='a')
        self.track_log  = Logger(cfg.LOG_DIR + '/tracking_log.txt', mode='a')


        # creat sesssion
        self.sess = tf.Session()
        self.use_pretrain_weights=[]
        self.build_net(top_shape)

        #init subnet
        self.tag=dist_name
        self.ckpt_dir = os.path.join(cfg.CHECKPOINT_DIR, dist_name) if weigths_dir == None else weigths_dir
        self.subnet_rpn=Net(prefix='MV3D', scope_name=network.top_view_rpn_name ,
                            checkpoint_dir=self.ckpt_dir)



        # set anchor boxes
        self.top_stride          = self.net.feature_stride_top
        top_feature_shape        = get_top_feature_shape(top_shape, self.top_stride)
        self.view_anchors_top, self.anchors_inside_inds = make_anchors(self.bases, self.top_stride, top_shape[0:2], top_feature_shape[0:2])
        self.anchors_inside_inds = np.arange(0, len(self.view_anchors_top), dtype=np.int32)  # use all

        self.log_subdir = None
        self.image_top  = None
        self.time_str   = None
        self.frame_info = None

        self.batch_top_inds         = None
        self.batch_top_labels       = None
        self.batch_top_pos_inds     = None
        self.batch_top_targets      = None
        self.batch_proposals        = None
        self.batch_proposal_scores  = None
        self.batch_gt_top_boxes     = None
        self.batch_gt_labels        = None


        # default_summary_writer
        self.default_summary_writer = None

        self.debug_mode = debug_mode

        # about tensorboard.
        self.tb_dir = dist_name if dist_name != None else strftime("%Y_%m_%d_%H_%M", localtime())


    def build_net(self, top_shape):
        with tf.variable_scope('MV3D'):

            self.net = load_net(top_shape, len(self.bases))

    def gc(self):
        self.log_subdir  = None
        self.image_top   = None
        self.time_str    = None
        self.frame_info  = None




    def keep_gt_inside_range(self, train_gt_labels, train_gt_boxes3d):
        if train_gt_labels.shape[0] == 0:
            return False, None, None


        assert train_gt_labels.shape[0] == train_gt_boxes3d.shape[0]
        # get limited train_gt_boxes3d and train_gt_labels.
        keep = np.zeros((len(train_gt_labels)), dtype=bool)

        for i in range(len(train_gt_labels)):
            if box3d_in_top_view(train_gt_boxes3d[i]):
                keep[i] = 1

        # if all targets are out of range in selected top view, return True.
        if np.sum(keep) == 0:
            return False, None, None

        train_gt_labels  = train_gt_labels[keep]
        train_gt_boxes3d = train_gt_boxes3d[keep]

        return True, train_gt_labels, train_gt_boxes3d




    def variables_initializer(self):
        self.sess.run(tf.global_variables_initializer(),
                 {IS_TRAIN_PHASE: True, K.learning_phase(): 1})


    def load_weights(self, weights=[]):
        for name in weights:
            if name == network.top_view_rpn_name:
                self.subnet_rpn.load_weights(self.sess)
            else:
                ValueError('unknow weigths name')

    def clean_weights(self, weights=[]):
        for name in weights:
            if name == network.top_view_rpn_name:
                self.subnet_rpn.clean_weights()
            else:
                ValueError('unknow weigths name')


    def save_weights(self, weights=[]):
        for name in weights:
            if name == network.top_view_rpn_name:
                self.subnet_rpn.save_weights(self.sess)
            else:
                ValueError('unknow weigths name')


    def image_top_padding(self, image_top):
        return np.concatenate((image_top, np.zeros_like(image_top)*255,np.zeros_like(image_top)*255), 1)


    def log_rpn(self,step=None, scope_name=''):

        image_top    = self.image_top
        subdir       = self.log_subdir
        top_inds     = self.batch_top_inds
        top_labels   = self.batch_top_labels
        top_pos_inds = self.batch_top_pos_inds

        top_targets     = self.batch_top_targets
        proposals       = self.batch_proposals
        proposal_scores = self.batch_proposal_scores
        gt_top_boxes    = self.batch_gt_top_boxes
        gt_labels       = self.batch_gt_labels

        if gt_top_boxes is not None:
            img_gt = draw_rpn_gt(image_top, gt_top_boxes, gt_labels)
            # nud.imsave('img_rpn_gt', img_gt, subdir)
            self.summary_image(img_gt, scope_name + '/img_rpn_gt', step=step)

        if top_inds is not None:
            img_label = draw_rpn_labels(image_top, self.view_anchors_top, top_inds, top_labels)
            # nud.imsave('img_rpn_label', img_label, subdir)
            self.summary_image(img_label, scope_name+ '/img_rpn_label', step=step)

        if top_pos_inds is not None:
            img_target = draw_rpn_targets(image_top, self.view_anchors_top, top_pos_inds, top_targets)
            # nud.imsave('img_rpn_target', img_target, subdir)
            self.summary_image(img_target, scope_name+ '/img_rpn_target', step=step)

        if proposals is not None:
            rpn_proposal = draw_rpn_proposal(image_top, proposals, proposal_scores)
            # nud.imsave('img_rpn_proposal', rpn_proposal, subdir)
            self.summary_image(rpn_proposal, scope_name + '/img_rpn_proposal',step=step)




    def summary_image(self, image, tag, summary_writer=None,step=None):

        if summary_writer == None:
            summary_writer=self.default_summary_writer

        im_summaries = []
        # Write the image to a string
        s = io.BytesIO()
        plt.imsave(s,image)

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=image.shape[0],
                                   width=image.shape[1])
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag=tag, image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        summary_writer.add_summary(summary, step)


    def summary_scalar(self, value, tag, summary_writer=None, step=None):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        if summary_writer == None:
            summary_writer=self.default_summary_writer
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        summary_writer.add_summary(summary, step)


class train(MV3D):

    def __init__(self, data_train, data_val, pre_weights, train_targets, dist_name=None,
                 continue_train=False, learning_rate=0.001):
        shape_top = data_train.get_shape()
        MV3D.__init__(self, shape_top, dist_name=dist_name)
        self.data_train             = data_train
        self.data_val               = data_val
        self.train_target           = train_targets
        self.train_summary_writer   = None
        self.val_summary_writer     = None
        self.tensorboard_dir        = None
        self.summ                   = None
        self.iter_debug             = 200
        self.n_global_step          = 0
        # saver
        with self.sess.as_default():

            with tf.variable_scope('minimize_loss'):
                # solver
                self.learning_rate = tf.placeholder(tf.float32, shape=[])
                solver = tf.train.AdamOptimizer(learning_rate=learning_rate)

                # summary
                self.cls_loss_top = self.net.cls_loss_top
                tf.summary.scalar('cls_loss_top', self.cls_loss_top)

                self.reg_loss_top = self.net.reg_loss_top
                tf.summary.scalar('reg_loss_top', self.reg_loss_top)


                train_var_list =[]

                assert train_targets != []
                for target in train_targets:
                    # variables
                    if target == top_view_rpn_name:
                        train_var_list += self.subnet_rpn.variables
                    else:
                        ValueError('unknow train_target name')

                # set loss
                if set([network.top_view_rpn_name]) == set(train_targets):
                    targets_loss = 1. * self.cls_loss_top + 0.05 * self.reg_loss_top
                else:
                    ValueError('unknow train_target set')



                tf.summary.scalar('targets_loss', targets_loss)
                self.solver_step = solver.minimize(loss = targets_loss,var_list=train_var_list)


            # summary.FileWriter
            train_writer_dir = os.path.join(cfg.LOG_DIR, 'tensorboard', self.tb_dir + '_train')
            val_writer_dir = os.path.join(cfg.LOG_DIR, 'tensorboard',self.tb_dir + '_val')
            if continue_train == False:
               if os.path.isdir(train_writer_dir):
                   command ='rm -rf %s' % train_writer_dir
                   print('\nClear old summary file: %s' % command)
                   os.system(command)
               if os.path.isdir(val_writer_dir):
                   command = 'rm -rf %s' % val_writer_dir
                   print('\nClear old summary file: %s' % command)
                   os.system(command)

            self.train_summary_writer = tf.summary.FileWriter(train_writer_dir,graph=tf.get_default_graph())
            self.val_summary_writer = tf.summary.FileWriter(val_writer_dir)

            summ = tf.summary.merge_all()
            self.summ = summ

            self.variables_initializer()

            #remove old weigths
            if continue_train == False:
                self.clean_weights(train_targets)

            self.load_weights(pre_weights)
            if continue_train: self.load_progress()


    def anchors_details(self):
        pos_indes = self.batch_top_pos_inds
        top_inds  = self.batch_top_inds
        return 'anchors: positive= {} total= {}\n'.format(len(pos_indes), len(top_inds))


    def rpn_poposal_details(self):
        top_rois = self.batch_top_rois[0]
        labels   = self.batch_fuse_labels[0]
        total    = len(top_rois)
        fp       = np.sum(labels == 0)
        pos      = total - fp
        info     = 'RPN proposals: positive= {} total= {}'.format(pos, total)
        return info




    def log_info(self, subdir, info):
        dir = os.path.join(cfg.LOG_DIR, subdir)
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'info.txt'), 'w') as info_file:
            info_file.write(info)

    def save_progress(self):
        print('Save progress !')
        path = os.path.join(cfg.LOG_DIR, 'train_progress',self.tag,'progress.data')
        os.makedirs(os.path.dirname(path) ,exist_ok=True)
        pickle.dump(self.n_global_step, open(path, "wb"))


    def load_progress(self):
        path = os.path.join(cfg.LOG_DIR, 'train_progress', self.tag, 'progress.data')
        if os.path.isfile(path):
            print('\nLoad progress !')
            self.n_global_step = pickle.load(open(path, 'rb'))
        else:
            print('\nCan not found progress file')


    def __call__(self, max_iter=1000, data_train =None, data_val =None):

        sess = self.sess
        net  = self.net
        with sess.as_default():
            #for init model

            batch_size=1

            validation_step = cfg.validation_step
            ckpt_save_step  = cfg.ckpt_save_step


            if cfg.TRAINING_TIMER:
                time_it = timer()

            # start training here  #########################################################################################
            self.log_msg.write('iter |  cls_loss_top   reg_loss_top |  \n')
            self.log_msg.write('---------------------------------------\n')

            for iter in range(max_iter):

                is_validation   = False
                summary_it      = False
                summary_runmeta = False
                print_loss      = False
                log_this_iter   = False

                # set fit flag
                if iter % validation_step == 0:  summary_it,is_validation,print_loss = True,True,True # summary validation loss
                if (iter+1) % validation_step == 0:  summary_it,print_loss = True,True # summary train loss
                if iter % 20 == 0: print_loss = True #print train loss

                if 1 and  iter%300 == 3: summary_it,summary_runmeta = True,True

                if iter % self.iter_debug == 0 or (iter + 1) % self.iter_debug == 0:
                    log_this_iter = True
                    print('Summary log image')
                    if iter % self.iter_debug == 0: is_validation =False
                    else: is_validation =True

                data_set = self.data_val if is_validation else self.data_train
                self.default_summary_writer = self.val_summary_writer if is_validation else self.train_summary_writer
                step_name = 'validation' if is_validation else 'training'

                # load dataset
                self.batch_view_top, self.batch_gt_labels, self.batch_gt_boxes3d, self.frame_id = \
                    data_set.load(batch_size, shuffled=True)


                is_gt_inside_range, batch_gt_labels_in_range, batch_gt_boxes3d_in_range = \
                    self.keep_gt_inside_range(self.batch_gt_labels[0], self.batch_gt_boxes3d[0])

                if not is_gt_inside_range: continue

                self.batch_gt_labels     = np.zeros((1, batch_gt_labels_in_range.shape[0]), dtype=np.int32)
                self.batch_gt_boxes3d    = np.zeros((1, batch_gt_labels_in_range.shape[0], 8, 3), dtype=np.float32)
                self.batch_gt_labels[0]  = batch_gt_labels_in_range
                self.batch_gt_boxes3d[0] = batch_gt_boxes3d_in_range


                # fit_iterate log init
                if log_this_iter:
                    self.time_str   = strftime("%Y_%m_%d_%H_%M", localtime())
                    self.frame_info = data_set.get_frame_info(self.frame_id)[0]
                    self.log_subdir = step_name + '/' + self.time_str
                    image_top = draw_image_top(self.batch_view_top[0])
                    self.image_top  = self.image_top_padding(image_top)

                # fit
                t_cls_loss, t_reg_loss= \
                    self.fit_iteration(self.batch_view_top,\
                                       self.batch_gt_labels, self.batch_gt_boxes3d, self.frame_id,\
                                       is_validation =is_validation, summary_it=summary_it,\
                                       summary_runmeta=summary_runmeta, log=log_this_iter)

                if print_loss:
                    self.log_msg.write('%10s: |  %5d  %0.5f   %0.5f |\n' % \
                                       (step_name, self.n_global_step, t_cls_loss, t_reg_loss))

                if iter%ckpt_save_step==0:
                    self.save_weights(self.train_target)


                    if cfg.TRAINING_TIMER:
                        self.log_msg.write('It takes %0.2f secs to train %d iterations. \n' % \
                                           (time_it.time_diff_per_n_loops(), ckpt_save_step))
                self.gc()
                self.n_global_step += 1


            if cfg.TRAINING_TIMER:
                self.log_msg.write('It takes %0.2f secs to train the dataset. \n' % \
                                   (time_it.total_time()))
            self.save_progress()



    def fit_iteration(self, batch_view_top, batch_gt_labels, batch_gt_boxes3d,\
                      frame_id, is_validation =False, summary_it=False, summary_runmeta=False, log=False):

        net  = self.net
        sess = self.sess
        # put tensorboard inside
        cls_loss_top = net.cls_loss_top
        reg_loss_top = net.reg_loss_top

        self.batch_gt_top_boxes = box3d_to_top_box(batch_gt_boxes3d[0])


        ## generate proposals
        fd1 = {
            net.view_top:          batch_view_top,
            net.anchors_top:       self.view_anchors_top,
            net.inside_inds_top:   self.anchors_inside_inds,

            IS_TRAIN_PHASE: True,
            K.learning_phase(): 1
        }

        self.batch_proposals, self.batch_proposal_scores, self.batch_features_top = \
            sess.run([net.proposals_top, net.proposal_scores_top, net.features_top], fd1)

        ## generate  train rois  for RPN
        self.batch_top_inds, self.batch_top_pos_inds, self.batch_top_labels, self.batch_top_targets = \
            rpn_target(self.view_anchors_top, self.anchors_inside_inds, batch_gt_labels[0],
                       self.batch_gt_top_boxes)

        # print ('self.batch_top_targets')
        # print (self.batch_top_targets)


        if log:
            step_name  = 'validation' if is_validation else  'train'
            scope_name = '%s_iter_%06d' % (step_name, self.n_global_step - (self.n_global_step % self.iter_debug))
            self.log_rpn(step=self.n_global_step, scope_name=scope_name)

        if log:
            log_info_str = 'frame info: ' + self.frame_info + '\n'
            log_info_str += self.anchors_details()
            self.log_info(self.log_subdir, log_info_str)


        ## run reg_loss_top and cls_loss_top
        fd2 = {
            **fd1,
            net.view_top: batch_view_top,

            net.inds_top: self.batch_top_inds,
            net.pos_inds_top: self.batch_top_pos_inds,

            net.labels_top: self.batch_top_labels,
            net.targets_top: self.batch_top_targets
        }



        if summary_it:
            run_options = None
            run_metadata = None

            if is_validation:
                t_cls_loss, t_reg_loss, tb_sum_val = \
                    sess.run([cls_loss_top, reg_loss_top, self.summ], fd2)
                self.val_summary_writer.add_summary(tb_sum_val, self.n_global_step)
                print('added validation  summary ')
            else:
                if summary_runmeta:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                _, t_cls_loss, t_reg_loss, tb_sum_val = \
                    sess.run([self.solver_step, cls_loss_top, reg_loss_top,
                              self.summ], feed_dict=fd2, options=run_options, run_metadata=run_metadata)
                self.train_summary_writer.add_summary(tb_sum_val, self.n_global_step)
                print('added training  summary ')

                if summary_runmeta:
                    self.train_summary_writer.add_run_metadata(run_metadata, 'step%d' % self.n_global_step)
                    print('added runtime metadata.')

        else:
            if is_validation:
                t_cls_loss, t_reg_loss = \
                    sess.run([cls_loss_top, reg_loss_top], fd2)
            else:

                _, t_cls_loss, t_reg_loss = \
                    sess.run([self.solver_step, cls_loss_top, reg_loss_top],
                             feed_dict=fd2)


        return t_cls_loss, t_reg_loss

