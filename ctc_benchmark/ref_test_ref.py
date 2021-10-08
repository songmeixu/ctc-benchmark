#!/usr/bin/env python

import argparse
import time

import mxnet as mx

def load_model(mxnet_json, mxnet_params):
    symbol = mx.sym.load(mxnet_json)
    save_dict = mx.nd.load(mxnet_params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return (symbol, arg_params, aux_params)

# Basic Conv + BN + ReLU factory
def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), act_type="relu"):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data = bn, act_type=act_type)
    return act

# A Simple Downsampling Factory
def DownsampleFactory(data, ch_3x3):
    # conv 3x3
    conv = ConvFactory(data=data, kernel=(3, 3), stride=(2, 2), num_filter=ch_3x3, pad=(1, 1))
    # pool
    pool = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max')
    # concat
    concat = mx.symbol.Concat(*[conv, pool])
    return concat

# A Simple module
def SimpleFactory(data, ch_1x1, ch_3x3):
    # 1x1
    conv1x1 = ConvFactory(data=data, kernel=(1, 1), pad=(0, 0), num_filter=ch_1x1)
    # 3x3
    conv3x3 = ConvFactory(data=data, kernel=(3, 3), pad=(1, 1), num_filter=ch_3x3)
    #concat
    concat = mx.symbol.Concat(*[conv1x1, conv3x3])
    return concat

def inception():
    data = mx.symbol.Variable(name="data")
    conv1 = ConvFactory(data=data, kernel=(3,3), pad=(1,1), num_filter=96, act_type="relu")
    in3a = SimpleFactory(conv1, 32, 32)
    in3b = SimpleFactory(in3a, 32, 48)
    in3c = DownsampleFactory(in3b, 80)
    in4a = SimpleFactory(in3c, 112, 48)
    in4b = SimpleFactory(in4a, 96, 64)
    in4c = SimpleFactory(in4b, 80, 80)
    in4d = SimpleFactory(in4c, 48, 96)
    in4e = DownsampleFactory(in4d, 96)
    in5a = SimpleFactory(in4e, 176, 160)
    in5b = SimpleFactory(in5a, 176, 160)
    pool = mx.symbol.Pooling(data=in5b, pool_type="avg", kernel=(7,7), name="global_pool")
    flatten = mx.symbol.Flatten(data=pool, name="flatten1")
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=10, name="fc1")
    softmax = mx.symbol.SoftmaxOutput(data=fc, name="loss")
    return softmax



def build_model():
    softmax = inception()
    batch_size = 1
    grad_req = 'write'

    texec = softmax.simple_bind(ctx=mx.gpu(),
                                data=(batch_size, 3, 224, 224),
                                grad_req=grad_req)
    return texec

def main():
    parser = argparse.ArgumentParser(
        description = 'Forward and backward computation benchmark')
    parser.add_argument('-j', '--json',
                        help = 'The network architecture in MXNet format')
    parser.add_argument('-p', '--params',
                        help = 'The binary parameter file in MXNet format')
    parser.add_argument('-d', '--data_shape',
                       help = 'Data shape: batch_size,channels,height,width')
    parser.add_argument('-g', '--gpu_id', type = int, default = -1,
                        help = 'GPU ID to use, default -1 for CPU')
    parser.add_argument('-s', '--step', default = 'both',
                        help = 'Which step to run: forward, backward or both'
                        ', default both')
    parser.add_argument('-i', '--iterations', type = int, default = 1,
                        help = 'The number of iterations to run, default 1')
    args = parser.parse_args()
    for x in [args.json, args.params, args.data_shape]:
        if x is None:
            parser.print_help()
            return

    symbol, arg_params, aux_params = load_model(args.json, args.params)
    if args.gpu_id < 0:
        device = 'cpu'
    else:
        device = 'gpu'
    data_shape = tuple([int(x) for x in args.data_shape.split(',')])
    step = args.step.lower()
    ## To benchmark models converted from Caffe models
    executor = symbol.simple_bind(mx.Context(device), data = data_shape)
    ## To benchmark the inception model defined in
    ## mxnet/tests/python/multi-node/common.py
##    executor = build_model()
    total_forward = 0
    total_backward = 0
    total = 0
    iterations = max(1, args.iterations)
    begin = time.time()
    for i in xrange(iterations):
        start = time.time() * 1000
        if step in ['forward', 'both']:
            executor.forward()
        forward_end = time.time() * 1000
        total_forward += forward_end - start
        if step in ['backward', 'both']:
            executor.backward()
        backward_end = time.time() * 1000
        total_backward += backward_end - forward_end
        total += backward_end - start
        print 'Iteration:', i + 1
        print '\t Forward time', forward_end - start, 'ms'
        print '\t Backward time', backward_end - forward_end, 'ms'
        print '\t Forward-Backward time', backward_end - start, 'ms'
    print 'Average Forward time:', total_forward / iterations, 'ms'
    print 'Average Backward time:', total_backward / iterations, 'ms'
    print 'Average Forward-Backward time:', total / iterations, 'ms'
    print 'Total time', total, 'ms'
    print 'End time', time.time() - begin

if __name__ == '__main__':
    main()
input
license
line
