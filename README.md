# BinaryTernary-MNIST-training

Code from https://github.com/DingKe/nn_playground.git

To train binary MNIST (with relu or binary_tanh):

```
python mnist_mlp.py -o train_mnist_binary -p binary #with binary_tanh
python mnist_mlp.py -o train_mnist_binary -p binary --relu #with relu
```

To train ternary MNIST (with relu or ternary_tanh):

```
python mnist_mlp.py -o train_mnist_ternary -p ternary #with ternary_tanh
python mnist_mlp.py -o train_mnist_ternary -p ternary --relu #with relu
```

To train full-precision MNIST:

```
python mnist_mlp.py -o train_mnist_ternary -p float
```

To evaluate performances and check loss/accuracy history:

```
python eval_mnist.py <TRAININGDIR>
```
