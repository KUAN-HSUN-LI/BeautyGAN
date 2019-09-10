from data_set import *
from loss import *
from util import *
from his_match import *
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
import os
import matplotlib.pyplot as plt
import time
class instance_norm():
    def __init__(self, name,epsilon=1e-06):
        self.epsilon  = epsilon
        self.name = name

    def __call__(self, x):
        return tf.contrib.layers.batch_norm(x,
            epsilon=self.epsilon,
            scope=self.name)

class spectral_norm():
    def __init__(self, name, u_name):
        self.name = name
        self.u_name = u_name
    def __call__(self, w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable(self.u_name, [1, w_shape[-1]], initializer = tf.truncated_normal_initializer(), 
                            trainable = False)
        u_hat = u
        v_hat = None
        for i in range(iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = self.l2_norm(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = self.l2_norm(u_)
        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = tf.reshape(w_norm, w_shape)
        return w_norm
    def l2_norm(self, v, eps=1e-12):
        return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


class BeautyGAN(object):
    def __init__(self, batch_size = 1, lr = 2e-4):
        self.batch_size = batch_size
        self.lr = lr
        self.sess = tf.Session()
        self.graph()
        self.saver = tf.train.Saver()
        print('graph bulid up.')

        print("img data creating")
        t1 = time.time()
        self.trainData_makeup = img_gen('makeup', 'train', batch_size)
        self.train_makeup_data = self.trainData_makeup.data_gen()
        self.trainData_Nmakeup = img_gen('Nmakeup', 'train', batch_size)        
        self.train_Nmakeup_data = self.trainData_Nmakeup.data_gen()
        t2 = time.time()
        print("img data created time : %.2fs" %(t2 - t1))
        
        print("seg data creating")
        self.trainSeg_makeup_lip = seg_gen('makeup', 'train', [feature.Ulip.value, feature.Dlip.value], batch_size)
        self.train_makeup_seg_lip = self.trainSeg_makeup_lip.data_gen()
        self.trainSeg_Nmakeup_lip = seg_gen('Nmakeup', 'train', [feature.Ulip.value, feature.Dlip.value], batch_size)
        self.train_Nmakeup_seg_lip = self.trainSeg_Nmakeup_lip.data_gen()
        t3 = time.time()
        print("seg data created time : %.2fs" %(t3 - t2))


    def my_Conv2d(self, input_tensor, filters, name, kernel_size = (4,4), strides = (2,2)):
        x = tf.layers.conv2d(input_tensor, filters, kernel_size, strides, padding="same", 
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return x
    
    
    def discriminator(self, name, input_tensor):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = input_tensor

            w1 = tf.get_variable("w1", shape=[4, 4, x.get_shape()[-1], 64])
            b1 = tf.get_variable("b1", [64], initializer = tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input = x, filter = spectral_norm("w_spec1", "u1")(w1), strides = [1, 2, 2, 1], padding='SAME') + b1
            x = tf.nn.leaky_relu(x, alpha=-0.01)

            w2 = tf.get_variable("w2", shape=[4, 4, x.get_shape()[-1], 128])
            b2 = tf.get_variable("b2", [128], initializer = tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input = x, filter = spectral_norm("w_spec2", "u2")(w2), strides = [1, 2, 2, 1], padding='SAME') + b2
            x = tf.nn.leaky_relu(x, alpha=-0.01)

            w3 = tf.get_variable("w3", shape=[4, 4, x.get_shape()[-1], 256])
            b3 = tf.get_variable("b3", [256], initializer = tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input = x, filter = spectral_norm("w_spec3", "u3")(w3), strides = [1, 2, 2, 1], padding='SAME') + b3
            x = tf.nn.leaky_relu(x, alpha=-0.01)

            w4 = tf.get_variable("w4", shape=[4, 4, x.get_shape()[-1], 512])
            b4 = tf.get_variable("b4", [512], initializer = tf.constant_initializer(0.0))
            x = tf.nn.conv2d(input = x, filter = spectral_norm("w_spec4", "u4")(w4), strides = [1, 1, 1, 1], padding='SAME') + b4
            x = tf.nn.leaky_relu(x, alpha=-0.01)

            x = self.my_Conv2d(x, 1, 'd_out',strides = (1,1))
            output = tf.nn.leaky_relu(x, alpha=-0.01)

            return output
        
    def generator_in(self, name, input_tensor):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = input_tensor

            x = self.my_Conv2d(x, 64, 'gen_in_0', (7,7))
            x = instance_norm('g_in1')(x)
            x = tf.nn.relu(x)

            x = self.my_Conv2d(x, 128, 'gen_in_1', (4,4))
            x = instance_norm('g_in2')(x)
            x = tf.nn.relu(x)
            return x
        
    def generator(self, input_tensor):
        def residual(input_tensor, filters, name):
            temp = input_tensor
            
            x = tf.layers.conv2d(input_tensor, filters, (3,3), (1,1), padding="same", 
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            x = instance_norm(name)(x)
            x = tf.nn.relu(x)
            x += temp
            return x
        
        with tf.variable_scope("g_concat", reuse=tf.AUTO_REUSE):
            x = self.my_Conv2d(input_tensor, 256, 'Down-sampling')
            x = instance_norm('g_con_in')(x)
            x = tf.nn.relu(x)
            
            x = residual(x, 256, 'residual_1')            
            x = residual(x, 256, 'residual_2')
            x = residual(x, 256, 'residual_3')
            x = residual(x, 256, 'residual_4')
            x = residual(x, 256, 'residual_5')
            x = residual(x, 256, 'residual_6')
            
            
            x = tf.layers.conv2d_transpose(x, 128, (4,4), (2,2), padding="same", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            x = instance_norm('g_con_in2')(x)

            x = tf.layers.conv2d_transpose(x, 128, (4,4), (2,2), padding="same", 
                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            x = instance_norm('g_con_in3')(x)

            x = tf.layers.conv2d_transpose(x, 64, (4,4), (2,2), padding="same", 
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
            x = instance_norm('g_con_in4')(x)
            return x
        
    def generator_out(self, name, input_tensor):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            x = input_tensor
            x = self.my_Conv2d(x, 64, 'gen_out_0', (3,3), (1,1))
            x = instance_norm('g_out0')(x)
            x = tf.nn.relu(x)

            x = self.my_Conv2d(x, 64, 'gen_out_1', (3,3), (1,1))
            x = instance_norm('g_out1')(x)
            x = tf.nn.relu(x)

            x = self.my_Conv2d(x, 3, 'gen_out_2', (7,7), (1,1))
            x = tf.nn.tanh(x)
            return x
    
    def VGG(self, name, input_tensor):
        with tf.variable_scope(name, reuse=True):
            



            vgg = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

            vgg.trainable = False
            style_layers = ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1', 
                            'block4_conv1']
            style_outputs = [vgg.get_layer(name).output for name in style_layers]
            return style_outputs

    def graph(self):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            self.ref_data      = tf.placeholder(tf.float32, shape=[self.batch_size, 256, 256, 3])
            self.src_data      = tf.placeholder(tf.float32, shape=[self.batch_size, 256, 256, 3])
            self.src_seg       = tf.placeholder(tf.float32, shape=[self.batch_size, 256, 256, 3])
            self.src_his_lip   = tf.placeholder(tf.float32, shape=[self.batch_size, 256, 256, 3])
            self.makeup_lambda = tf.placeholder(tf.float32, shape=[1])
            
            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                self.g1_ref = self.generator_in('g_in_ref', self.ref_data)
                self.g1_src = self.generator_in('g_in_src', self.src_data)
                self.g1_concat = tf.concat([self.g1_ref, self.g1_src], -1)
                self.g1_concat = self.generator(self.g1_concat)
                self.g1_ref_A = self.generator_out('g_out_ref_A', self.g1_concat)
                self.g1_src_B = self.generator_out('g_out_src_B', self.g1_concat)
                self.g2_ref = self.generator_in('g_in_ref', self.g1_ref_A)
                self.g2_src = self.generator_in('g_in_src', self.g1_src_B)
                self.g2_concat = tf.concat([self.g2_ref, self.g2_src], -1)
                self.g2_concat = self.generator(self.g2_concat)
                self.g2_ref_rec = self.generator_out('g_out_ref_A', self.g2_concat)
                self.g2_src_rec = self.generator_out('g_out_src_B', self.g2_concat)
            with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                with tf.variable_scope("discriminatorA", reuse=tf.AUTO_REUSE):
                    self.src_logits = self.discriminator('discriminator_A', self.src_data)
                    self.ref_A_logits = self.discriminator('discriminator_A', self.g1_ref_A)
                with tf.variable_scope("discriminatorB", reuse=tf.AUTO_REUSE):               
                    self.ref_logits = self.discriminator('discriminator_B', self.ref_data)
                    self.src_B_logits = self.discriminator('discriminator_B', self.g1_src_B)


            self.VGG_src_B = tf.keras.applications.vgg16.preprocess_input((self.g1_src_B+1)/2 * 255)
            self.VGG_src =  tf.keras.applications.vgg16.preprocess_input((self.src_data+1)/2 * 255)
            self.VGG_ref_A =  tf.keras.applications.vgg16.preprocess_input((self.g1_ref_A+1)/2 * 255)
            self.VGG_ref =  tf.keras.applications.vgg16.preprocess_input((self.ref_data+1)/2 * 255)


            self.src_style   = self.VGG('vgg', tf.concat([self.VGG_src_B,  self.VGG_src], axis = 0))
            self.ref_style   = self.VGG('vgg', tf.concat([self.VGG_ref_A,  self.VGG_ref], axis = 0))

            g_vars = tf.trainable_variables('generator')
            dA_vars = tf.trainable_variables("discriminator/discriminatorA")
            dB_vars = tf.trainable_variables("discriminator/discriminatorB")            

            #loss function#
            self.DA_Loss, self.DB_Loss = discrimination_loss(self.src_logits, self.src_B_logits, self.ref_logits, self.ref_A_logits)
            self.LDA, self.LDB = generation_loss(self.src_logits, self.src_B_logits, self.ref_logits, self.ref_A_logits)
            self.disLoss       = self.LDA + self.LDB
            self.cycleLoss     = cycle_consistency_loss(self.src_data, self.g2_src_rec, self.ref_data, self.g2_ref_rec)
            src_perLoss        = styleloss(self.src_style)
            ref_perLoss        = styleloss(self.ref_style)
            self.perLoss       = src_perLoss + ref_perLoss
            
            self.makeup_loss   = makeup_loss(self.g1_src_B*self.src_seg, self.src_his_lip, self.makeup_lambda, 1)
            
            self.Loss     = self.perLoss * 0.005 + self.disLoss + 10 * self.cycleLoss + self.makeup_loss
            gen_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr, beta1=0.9, beta2=0.99)
            disA_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr, beta1=0.9, beta2=0.99)
            disB_optimizer = tf.train.AdamOptimizer(learning_rate = self.lr, beta1=0.9, beta2=0.99)
            
            self.gen_train_op = gen_optimizer.minimize(self.Loss,var_list=g_vars)
            self.disA_train_op = disA_optimizer.minimize(self.DA_Loss,var_list=dA_vars)
            self.disB_train_op = disB_optimizer.minimize(self.DB_Loss,var_list=dB_vars)
            self.sess.run(tf.global_variables_initializer())

            
        writer = tf.summary.FileWriter("TensorBoard/", graph = self.sess.graph)
    def train(self, epochs):
        print("train start")
        def brief(f):
            return "{:.5f}".format(f)
        for i in range(epochs):
            start = time.time()
            loss  = 0
            for j in range(1001):
                
                src_img = next(self.train_Nmakeup_data)
                ref_img = next(self.train_makeup_data)
                src_seg = next(self.train_Nmakeup_seg_lip)
                ref_seg = next(self.train_makeup_seg_lip)
                src = src_img*src_seg
                ref = ref_img*ref_seg
                ref = ((ref+1)/2 * 255).astype(np.int16)
                src = ((src+1)/2 * 255).astype(np.int16)
                his_feature = hist_match(src[0], ref[0], 5)
                his_feature = (his_feature.astype(np.float32) - 128)/128.
                his_feature = np.reshape(his_feature, (1,256,256,3))
                lambda_list = np.array([1.0])


                feed_dict1 = {self.ref_data : ref_img, self.src_data : src_img}
                feed_dict2 = {self.ref_data : ref_img, self.src_data : src_img,
                              self.src_seg  : src_seg, self.src_his_lip : his_feature, 
                              self.makeup_lambda : lambda_list}
                LDA       = self.sess.run(self.DA_Loss, feed_dict = feed_dict1)

                LDB       = self.sess.run(self.DB_Loss, feed_dict = feed_dict1)

                self.sess.run([self.disA_train_op, self.disB_train_op], feed_dict = feed_dict1)
                self.sess.run(self.gen_train_op, feed_dict = feed_dict2)
                Loss, src_rec, ref_rec, src_B, ref_A, makeup_loss = self.sess.run([self.Loss, 
                                                                self.g2_src_rec , 
                                                                self.g2_ref_rec , 
                                                                self.g1_src_B ,
                                                                self.g1_ref_A, self.makeup_loss], feed_dict = feed_dict2)
                print("batch:", j ," LDA:", brief(LDA), "LDB:", brief(LDB), "makeup:", brief(makeup_loss), "Loss:", brief(Loss))
                if j % 100 == 0:
                    src_img = ((src_img+1)/2 * 255).astype(np.uint8)
                    ref_img = ((ref_img+1)/2 * 255).astype(np.uint8)
                    src_rec = ((src_rec+1)/2 * 255).astype(np.uint8)
                    ref_rec = ((ref_rec+1)/2 * 255).astype(np.uint8)
                    src_B   = ((src_B+1)/2 * 255).astype(np.uint8)
                    ref_A   = ((ref_A+1)/2 * 255).astype(np.uint8)
                    his_feature = ((his_feature+1)/2 * 255).astype(np.uint8)
                    plt.imsave('./src_raw/epoch'+str(i)+'-'+str(j)+'.png', src_img[0])
                    plt.imsave('./ref_raw/epoch'+str(i)+'-'+str(j)+'.png', ref_img[0])
                    plt.imsave('./src_rec/epoch'+str(i)+'-'+str(j)+'.png', src_rec[0])
                    plt.imsave('./ref_rec/epoch'+str(i)+'-'+str(j)+'.png', ref_rec[0])
                    plt.imsave('./src_B/epoch'  +str(i)+'-'+str(j)+'.png', src_B[0])
                    plt.imsave('./ref_A/epoch'  +str(i)+'-'+str(j)+'.png', ref_A[0])
                    plt.imsave('./his_feature/epoch'  +str(i)+'-'+str(j)+'.png', his_feature[0])
                    
                loss += Loss
            self.save_model("./model/epoch"+str(i)+"/beautygan.ckpt")
            end = time.time()
            print('epoch'+str(i)+' time:', end-start)
            print('epoch'+str(i)+' loss:', loss / 1001)
    def test(self):
        print("test start")
        def brief(f):
            return "{:.4f}".format(f)

    def save_model(self,path):
        self.saver.save(self.sess, path)
if __name__ == '__main__':
    if not os.path.exists('src_raw'):
        os.mkdir('src_raw', 0o755)
    if not os.path.exists('ref_raw'):
        os.mkdir('ref_raw', 0o755)
    if not os.path.exists('src_rec'):
        os.mkdir('src_rec', 0o755)
    if not os.path.exists('ref_rec'):
        os.mkdir('ref_rec', 0o755)
    if not os.path.exists('src_B'):
        os.mkdir('src_B', 0o755)
    if not os.path.exists('ref_A'):
        os.mkdir('ref_A', 0o755)
    if not os.path.exists('his_feature'):
        os.mkdir('his_feature', 0o755)
    if not os.path.exists('model'):
        os.mkdir('model' , 0o755)
    BeautyGAN = BeautyGAN()
    BeautyGAN.train(100)
