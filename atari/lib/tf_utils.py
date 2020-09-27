import tensorflow as tf

def dense_nn(inputs, layers_size, name = "mlp") :
    """
     Create densely connected multi-layer neural networks: 
     inputs: the input tensor
     layers_size: list of integers for size of each layer
     outputs: output with size layers_size[-1]
    """
    with tf.variable_scope(name) :
        for i, size in enumerate(layers_size) :
            outputs = tf.layers.dense(
                inputs, 
                size, 
                activation = tf.nn.relu if i < len(layers_size) -1 else None,
                kernel_initializer = tf.contrib.layers.xavier_initializer(),
                name = name + "_l" + str(i)
            )
            inputs = outputs
    return inputs