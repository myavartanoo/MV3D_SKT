import training     as mv3d
import config_rpn   as cfg
import argparse
from   batch_loading_rpn import batch_loading


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')

    all= '%s,%s,%s' % (mv3d_net.top_view_rpn_name ,mv3d_net.imfeature_net_name,mv3d_net.fusion_net_name)

    parser.add_argument('-w', '--weights', type=str, nargs='?', default=all,
        help='use pre trained weights example: -w "%s" ' % (all))

    parser.add_argument('-t', '--targets', type=str, nargs='?', default=all,
        help='train targets example: -w "%s" ' % (all))

    parser.add_argument('-i', '--max_iter', type=int, nargs='?', default=100000,
                        help='max count of train iter')

    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')

    parser.add_argument('-c', '--continue_train', type=bool, nargs='?', default=False,
                        help='set continue train flag')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' %tag)

    max_iter = args.max_iter
    weights=[]
    if args.weights != '':
        weights = args.weights.split(',')

    targets=[]
    if args.targets != '':
        targets = args.targets.split(',')

    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR


    cfg.DATA_SETS_TYPE == 'SKT'
    if cfg.OBJ_TYPE == 'car':
        train_car = 'object3d_all'
        train_data = 'all_val'
        train_dataset = {
            train_car: [train_data]
        }
    if cfg.OBJ_TYPE == 'car':
        validation_car = 'object3d_all'
        validation_data = 'all_val'
        validation_dataset = {
            validation_car: [validation_data]
        }


    dataset_loader_train = batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR, train_dataset, is_testset=False)
    dataset_loader_validation = batch_loading(cfg.PREPROCESSED_DATA_SETS_DIR, validation_dataset, is_testset=False)


    train = mv3d.Trainer(train_set=dataset_loader_train, validation_set=dataset_loader_validation,
                         pre_trained_weights=weights, train_targets=targets, log_tag=tag,
                         continue_train=args.continue_train, learning_rate = 0.001)

    train(max_iter=max_iter)
