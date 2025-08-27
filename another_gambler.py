import numpy as np
import tensorflow as tf
# from tensorflow.keras.layers import Conv1D, GRU,SimpleRNN,GRU
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit/libdevice'

from tqdm import tqdm
tf.random.set_seed(
    384
)

# Conv1D=
with tf.device("GPU:0"):
    def gen(n,batch):
        inputs=tf.keras.layers.Input(shape=(150,5),batch_size=batch)
        inputs2=tf.keras.layers.Input(shape=(150,1),batch_size=batch)
        

        x1=tf.keras.layers.Conv1D(n*4,kernel_size=1,strides=1,activation="elu",dilation_rate=128)(inputs)
        x2=tf.keras.layers.Conv1D(n*4,kernel_size=1,strides=1,activation="elu",dilation_rate=64)(inputs)
        x3=tf.keras.layers.Conv1D(n*4,kernel_size=1,strides=1,activation="elu",dilation_rate=64)(inputs)
        x4=tf.keras.layers.Conv1D(n*4,kernel_size=1,strides=1,activation="elu",dilation_rate=36)(inputs)
        x5=tf.keras.layers.Conv1D(n*4,kernel_size=1,strides=1,activation="elu",dilation_rate=18)(inputs2)
        x6=tf.keras.layers.Conv1D(n*4,kernel_size=1,strides=1,activation="elu",dilation_rate=9)(inputs2)
        
        bypass,S,_0=tf.keras.layers.LSTM(n//2,return_sequences=True,return_state=True)(inputs2)
        
        x1,N1,_1=tf.keras.layers.LSTM(n//2,return_sequences=True,return_state=True)(x1,initial_state=[S,_0])
        x11=tf.keras.layers.Conv1D(n//2,kernel_size=1,strides=1,activation="elu",dilation_rate=32)(x1)
        x2,N2,_2=tf.keras.layers.LSTM(n//2,return_sequences=True,return_state=True)(x2,initial_state=[N1,_1])
        x21=tf.keras.layers.Conv1D(n//2,kernel_size=1,strides=1,activation="elu",dilation_rate=32)(x2)
        x3,N3,_3=tf.keras.layers.LSTM(n//2,return_sequences=True,return_state=True)(x3,initial_state=[N2,_2])
        x31=tf.keras.layers.Conv1D(n//2,kernel_size=1,strides=1,activation="elu",dilation_rate=32)(x3)
        x4,N4,_4=tf.keras.layers.LSTM(n//2,return_sequences=True,return_state=True)(x4,initial_state=[N3,_3])
        x41=tf.keras.layers.Conv1D(n//2,kernel_size=1,strides=1,activation="elu",dilation_rate=32)(x4)
        
        x5,N5,_5=tf.keras.layers.LSTM(n//2,return_sequences=True,return_state=True)(x5,initial_state=[N4,_4])
        x51=tf.keras.layers.Conv1D(n//2,kernel_size=1,strides=1,activation="elu",dilation_rate=16)(x5)
        x6,N6,_6=tf.keras.layers.LSTM(n//2,return_sequences=True,return_state=True)(bypass,initial_state=[N5,_5])
        x61=tf.keras.layers.Conv1D(n//2,kernel_size=1,strides=1,activation="elu",dilation_rate=16)(bypass)
        x7,N7,_7=tf.keras.layers.LSTM(n//2,return_sequences=True,return_state=True)(x6,initial_state=[N6,_6])
        x71=tf.keras.layers.Conv1D(n//2,kernel_size=1,strides=1,activation="elu",dilation_rate=8)(x6)
        
        sum1=tf.keras.layers.Add()([x1,x2,x3,x4,x11,x21,x31,x41])
        sum2=tf.keras.layers.Add()([x5,x6,x7,x51,x61,x71])
        
        d1=tf.keras.layers.MaxPooling1D(pool_size=1,strides=155)(sum1)
        d2=tf.keras.layers.MaxPooling1D(pool_size=1,strides=155)(sum2)
        
        final=tf.keras.layers.Conv1D(5,kernel_size=1,strides=1,activation="elu",dilation_rate=32)(d1)
        final2=tf.keras.layers.Conv1D(1,kernel_size=1,strides=1,activation="elu",dilation_rate=32)(d2)
        
        return tf.keras.Model([inputs,inputs2],[final,final2])

    model_generator=gen(300,1)
    
    model_generator.compile()
    model_generator.summary()

    generator_optimizer = tf.keras.optimizers.AdamW(1e-6)
    
    def discriminator_loss(inp1,inp2,pred1,pred2):
        no=5
        diff1=0.0
        diff2=0.0
        no2=1
        # print(inp1[0])
        for i in range(len(inp1[0])):
            for j in range(len(inp1[0,0])):
                if inp1[0,i,j]-0.5<=pred1[0,i,j]<=inp1[0,i,j]+0.5:
                    no-=1
        for i in range(len(inp1[0])):
            for j in range(len(inp1[0,0])):
                diff1+=((inp1[0,i,j]-pred1[0,i,j])**2)*1e10
        for i in range(len(inp2[0])):
            for j in range(len(inp2[0,0])):
                if inp2[0,i]-0.1<=pred2[0,i]<=inp2[0,i]+0.1:
                    no2-=1
        for i in range(len(inp1[0])):
            for j in range(len(inp2[0,0])):
                diff2+=((inp2[0,i]-pred2[0,i])**2)*1e10*no2
        if no>2:
            diff1-=1e10*no**4
        if no2<=1:
            diff2-=9e12*no
        if no>=4 and no2<1:
            diff1-=1e14
            
        return diff1+diff2
            
        # return diff1+diff2

    # def generator_loss(fake_output,real):
    #     return tf.keras.losses.binary_crossentropy(real, fake_output)

    mse = tf.keras.losses.MeanSquaredError()

    data=np.loadtxt("ep.txt",dtype=np.int16)
    data=np.round(data)
    inp1=data[:,:5]
    inp2=data[:,5][:, np.newaxis]
    # print()
    inp1=np.stack((inp1,)*1,axis=0,dtype=np.int16)
    inp2=np.stack((inp2,)*1,axis=0,dtype=np.int16)
    print(np.shape(inp1),np.shape(inp2))
    inp1=tf.convert_to_tensor(inp1,dtype=tf.float16)
    inp2=tf.convert_to_tensor(inp2,dtype=tf.float16)
    def train_step(input1,input2, comparable1,comparable2):
        
        
        with tf.GradientTape() as gen_tape:
            generated_images = model_generator([input1,input2], training=True)
            # print(generated_images)
            gen_mse=discriminator_loss(comparable1,comparable2,generated_images[0],generated_images[1])
            # gen_mse=mse(comparable1,generated_images[0])+mse(comparable2,generated_images[1])

        gradients_gen = gen_tape.gradient(gen_mse, model_generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_gen, model_generator.trainable_variables))
        # gen_loss_float = float(gen_loss.numpy())
        return gen_mse.numpy()
    def train(dataset1,dataset2, epochs):
        best_disc_acc = float('inf')  # Initialize with infinity for comparison

        for epoch in range(epochs):
            
            epoch_gen_loss = 0.0
            fake_mse=0.0
            
            for i in tqdm(range(len(dataset1[0])-150)):
                
                inputs1 = dataset1[:, i:i+150, :]       # First 5 timesteps
                inputs2=dataset2[:, i:i+150, :] 
                comparable1 = np.stack((dataset1[:, i+150, :],)*1,axis=0)
                comparable2 = np.stack((dataset2[:, i+150, :],)*1,axis=0)
                gen_loss = train_step(inputs1, inputs2, comparable1, comparable2)  # No unpacking
                epoch_gen_loss += gen_loss
                fake_mse += gen_loss  # If this is the intended logic

                
            print(f"Epoch {epoch + 1}, Discriminator Loss = {fake_mse}")
            model_generator.save("ep_gen_chckpnt.keras")
            
            if fake_mse<best_disc_acc:
                best_disc_acc = fake_mse
                model_generator.save("GENERATOR_EP.keras")
                print(f"Discriminator model saved with loss {best_disc_acc}")

    train(inp1,inp2,50000)
