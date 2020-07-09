import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), 'input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), 'input_z')

    return inputs_real, inputs_z

def generator(z, out_dim, n_units=128 , reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        h1 = tf.layers.dense(z, n_units, activation=None)
        h1 = tf.maximum(alpha * h1, h1) #Leaky Relu

        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)

        return out

def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        h1 = tf.layers.dense(x, n_units, activation=None)
        h1 = tf.maximum(alpha * h1, h1)

        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)

        return  out, logits

# ハイパーパラメーターの初期化

input_size = 784
z_size = 100
g_hidden_size = 128
d_hidden_size = 128
alpha = 0.01
smooth = 0.1

tf.reset_default_graph()
input_real, input_z = model_inputs(input_size, z_size)

g_model = generator(input_z, input_size, n_units=g_hidden_size, alpha=alpha)

d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, n_units=d_hidden_size, alpha=alpha)

# Discriminator 損失関数の定義
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)*(1-smooth)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake

# Generator 損失関数の定義
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

# 最適化の定義
learning_rate = 0.002
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

batch_size = 100

epochs = 100
samples = []
losses = []

saver = tf.train.Saver(var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # reset
    for e in range(epochs):
        for i in range(mnist.train.num_examples//batch_size): # 各エポックのループ内でミニバッチ学習を行う回数 = サンプル総数/バッチサイズ
            batch = mnist.train.next_batch(batch_size)  # ミニバッチ

            batch_images = batch[0].reshape((batch_size, 784))  # batch_size(100行)*(784列)のデータセット
            batch_images = batch_images * 2 - 1  # 0~1の濃淡データを-1~1の値に変換。generatorから来るデータとrangeを揃える

            # Generator
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))  # 一様分布

            # トレーニングを実行
            _ = sess.run(d_train_optimize, feed_dict={input_real: batch_images, input_z: batch_z})  # 最適化計算・パラメータ更新を行う
            _ = sess.run(g_train_optimize, feed_dict={input_z: batch_z})  # 最適化計算・パラメータ更新を行う
            # _ =は、実行はするけど、値を保持しないときに使う

        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})  # トレーニングのロスを計算
        train_loss_g = g_loss.eval({input_z: batch_z})

        print("エポック {}/{} ".format(e+1,epochs),
            "D ロス: {:.4f}".format(train_loss_d),
            "G ロス: {:.4f}".format(train_loss_g))

        losses.append((train_loss_d,train_loss_g))

        sample_z = np.random.uniform(-1, 1, size=(16,z_size))
        gen_samples = sess.run(generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha), feed_dict ={input_z:sample_z})  # 画像ファイルの生成
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')  # 途中経過を保存

with open('training_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='D')  # Dロスのみを取得
plt.plot(losses.T[1], label='G')  # Gロスのみを取得
plt.title('Train Loss')
plt.legend()
plt.show()

#10エポックおきに、6個のデータを表示させる
rows,cols = 10, 6
#fig:グラフ全体、axes:個別指定表示要素
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes): #サンプル数/10おき
    #100/6 = 15こおきに取り出す
    for img, ax in zip(sample[::int(len(samples)/cols)],ax_row):#::x x分だけインクリメント 指定幅でデータをsamplesから取り出す
        #数字の列を濃淡差として扱い、画像として表示する
        #784個の1次元ベクトルを28*28の2次元行列に変換
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)