import os.path  as osp
import numpy    as np

from easydict import EasyDict as edict

__C = edict()
cfg = __C
__C.TEST_KEY=11

__C.DATA_SETS_TYPE='SKT'
__C.SINGLE_CLASS_DETECTION = True
__C.OBJ_TYPE = 'car' #'car' 'ped'

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))


__C.DATA_SETS_DIR=osp.join(__C.ROOT_DIR, 'data')

__C.RAW_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'raw', __C.DATA_SETS_TYPE)
__C.PREPROCESSED_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'preprocessed', __C.DATA_SETS_TYPE)
__C.PREPROCESSING_DATA_SETS_DIR = osp.join(__C.DATA_SETS_DIR, 'preprocessing', __C.DATA_SETS_TYPE)
__C.PREDICTED_XML_DIR = osp.join(__C.DATA_SETS_DIR, 'predicted', __C.DATA_SETS_TYPE)

__C.CHECKPOINT_DIR=osp.join(__C.ROOT_DIR,'checkpoint')
__C.LOG_DIR=osp.join(__C.ROOT_DIR,'log')

__C.TOP_CONV_KERNEL_SIZE = 3   #default 3

__C.RGB_BASENET = 'VGG'  # 'resnet' 'xception' 'VGG'

__C.validation_step=40
__C.ckpt_save_step=200


# config for lidar to top
__C.TOP_Y_MIN = 0
__C.TOP_Y_MAX = 512
__C.TOP_X_MIN = 0
__C.TOP_X_MAX = 512
__C.TOP_Z_MIN = 0
__C.TOP_Z_MAX = 36



# if timer is needed.
__C.TRAINING_TIMER = True
__C.TRACKING_TIMER = True
__C.DATAPY_TIMER = False

__C.USE_CLIDAR_TO_TOP = True

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value

if __name__ == '__main__':
    print('__C.ROOT_DIR = '+__C.ROOT_DIR)
    print('__C.DATA_SETS_DIR = '+__C.DATA_SETS_DIR)
    print('__C.RAW_DATA_SETS_DIR = '+__C.RAW_DATA_SETS_DIR)