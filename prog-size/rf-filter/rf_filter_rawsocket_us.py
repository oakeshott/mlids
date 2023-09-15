import os
import sys
import ctypes as ct
from ctypes import CDLL, POINTER
from ctypes import c_size_t, c_int32
import json
import numpy as np
curdir = os.path.dirname(__file__)

def ensure_contiguous(array):
    return np.ascontiguousarray(array) if not array.flags['C_CONTIGUOUS'] else array
if __name__ == '__main__':
    prefix_path = f"{curdir}/runs"
    num_estimators = 10
    children_left = []
    children_right = []
    threshold = []
    feature = []
    value = []
    for i in range(0, num_estimators):
        with open(f'{prefix_path}/childrenLeft{i}', 'r') as f:
            children_left.append(json.load(f))
        with open(f'{prefix_path}/childrenRight{i}', 'r') as f:
            children_right.append(json.load(f))
        with open(f'{prefix_path}/threshold{i}', 'r') as f:
            threshold.append(json.load(f))
        with open(f'{prefix_path}/feature{i}', 'r') as f:
            feature.append(json.load(f))
        with open(f'{prefix_path}/value{i}', 'r') as f:
            value.append(json.load(f))
    max_children_left_size  = max([len(children_left[i]) for i in range(num_estimators)])
    max_children_right_size = max([len(children_right[i]) for i in range(num_estimators)])
    max_feature_size        = max([len(feature[i]) for i in range(num_estimators)])
    max_value_size          = max([len(value[i]) for i in range(num_estimators)])
    max_threshold_size      = max([len(threshold[i]) for i in range(num_estimators)])
    for i in range(num_estimators):
        if len(value[i]) < max_value_size:
            tmp = max_value_size - len(value[i])
            value[i] += [0] * tmp
        if len(children_left[i]) < max_children_left_size:
            tmp = max_children_left_size - len(children_left[i])
            children_left[i] += [-1] * tmp
        if len(children_right[i]) < max_children_right_size:
            tmp = max_children_right_size - len(children_right[i])
            children_right[i] += [-1] * tmp
        if len(threshold[i]) < max_threshold_size:
            tmp = max_threshold_size - len(threshold[i])
            threshold[i] += [-2] * tmp
        if len(feature[i]) < max_feature_size:
            tmp = max_feature_size - len(feature[i])
            feature[i] += [-2] * tmp

    children_right = np.array(children_right).ravel()
    children_left  = np.array(children_left).ravel()
    threshold      = np.array(threshold).ravel()
    feature        = np.array(feature).ravel()
    value          = np.array(value).ravel()

    # value = np.array(value)
    # print(value)
    lib = CDLL(f"{curdir}/rf_filter.so")

    rf_filter = lib.rf_filter
    seconds = 110

    # ND_POINTER_1 = np.ctypeslib.ndpointer(dtype=children_left.dtype, ndim=1, flags="C_CONTIGUOUS")
    children_left_pointer  = children_left.ctypes.data_as(POINTER(ct.c_longlong))
    children_right_pointer = children_right.ctypes.data_as(POINTER(ct.c_longlong))
    threshold_pointer      = threshold.ctypes.data_as(POINTER(ct.c_longlong))
    feature_pointer        = feature.ctypes.data_as(POINTER(ct.c_longlong))
    value_pointer          = value.ctypes.data_as(POINTER(ct.c_longlong))

    c_uint_p = ct.POINTER(ct.c_uint)
    ret = ensure_contiguous(np.zeros(seconds, dtype=np.uintc))
    _ret = ret.ctypes.data_as(c_uint_p)

    rf_filter.argstypes = [c_int32, POINTER(ct.c_longlong), POINTER(ct.c_longlong), POINTER(ct.c_longlong), POINTER(ct.c_longlong), POINTER(ct.c_longlong), c_size_t]
    # rf_filter.argstypes = [ND_POINTER_1, c_size_t]
    rf_filter.restype = None

    # print(rf_filter(children_left))
    rf_filter(seconds, children_left_pointer, children_right_pointer, value_pointer, feature_pointer, threshold_pointer, max_children_right_size, _ret)
    # print(rf_filter(children_left, children_left.size))

    resdir = sys.argv[2]
    filename = f"{resdir}/rxpps.log"
    with open (filename, "w") as f:
        for d in np.ctypeslib.as_array(ret, seconds):
            f.write(f"{d}\n")
