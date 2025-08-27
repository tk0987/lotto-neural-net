import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Set CUDA path for XLA
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit/libdevice'

# Set random seed for reproducibility
tf.random.set_seed(384)

def gen(n, batch):
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

def generator_loss(inp1, inp2, pred1, pred2):
    mask_close = tf.logical_and(pred1 >= inp1 - 0.5, pred1 <= inp1 + 0.5)
    no = 5 - tf.reduce_sum(tf.cast(mask_close, tf.float32))

    diff1 = tf.reduce_sum(tf.where(pred1 < 0.8, 1.0 / (pred1 + 1e-12), 0.0))
    diff1 += tf.reduce_sum(tf.where(pred1 > 35, 100.0 * (pred1 + 1e-12), 0.0))
    diff1 += tf.reduce_sum(tf.square(inp1 - pred1))

    mask_close2 = tf.logical_and(pred2 >= inp2 - 0.1, pred2 <= inp2 + 0.1)
    no2 = 1 - tf.reduce_sum(tf.cast(mask_close2, tf.float32))

    diff2 = tf.reduce_sum(tf.where(pred2 < 0.8, 1.0 / (pred2 + 1e-12), 0.0))
    diff2 += tf.reduce_sum(tf.where(pred2 > 4.0, 100.0 * (pred2 + 1e-12), 0.0))
    diff2 += tf.reduce_sum(tf.square(inp2 - pred2)) * 1e10 * no2

    # Penalties
    diff1 = tf.where(no > 2, diff1 - 1e10 * tf.pow(no, 4), diff1)
    diff2 = tf.where(no2 <= 1, diff2 - 9e12 * no, diff2)
    diff1 = tf.where(tf.logical_and(no >= 4, no2 < 1), diff1 - 1e14, diff1)

    return diff1 + diff2

data = np.loadtxt("ep.txt", dtype=np.int16)
data = np.round(data)

with tf.device("GPU:0"):
    generators_main = [gen(256, 1) for _ in range(35)]
    generators_bonus = [gen(256, 1) for _ in range(4)]
    optimizers_main = [tf.keras.optimizers.AdamW(1e-4) for _ in range(35)]
    optimizers_bonus = [tf.keras.optimizers.AdamW(1e-4) for _ in range(4)]

def train_step(model, optimizer, inp_main, inp_bonus, target_main, target_bonus):
    with tf.GradientTape() as tape:
        pred_main, pred_bonus = model([inp_main, inp_bonus], training=True)
        loss = generator_loss(target_main, target_bonus, pred_main, pred_bonus)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, pred_main, pred_bonus

def train(dataset, epochs):
    best_loss_main = [float('inf')] * 35
    best_loss_bonus = [float('inf')] * 4
    total_iters = len(dataset) - 201

    for epoch in range(epochs):
        for i in tqdm(range(total_iters), desc=f"Epoch {epoch+1}/{epochs}"):
            inp_main = dataset[i:i+150, :5][np.newaxis, ...]
            inp_bonus = dataset[i:i+150, 5:][np.newaxis, ...]
            target_main = dataset[i+150:i+151, :5][np.newaxis, ...]
            target_bonus = dataset[i+150:i+151, 5:][np.newaxis, ...]

            active_main = set()
            active_bonus = set()

            for feature_index in range(5):
                j = int(inp_main[0, 0, feature_index]) - 1
                model = generators_main[j]
                optimizer = optimizers_main[j]
                loss, _, _ = train_step(model, optimizer, inp_main, inp_bonus, target_main, target_bonus)
                active_main.add(j)

            bonus_index = int(inp_bonus[0, 0, 0]) - 1
            model = generators_bonus[bonus_index]
            optimizer = optimizers_bonus[bonus_index]
            loss, _, _ = train_step(model, optimizer, inp_main, inp_bonus, target_main, target_bonus)
            active_bonus.add(bonus_index)

        # Save best models
        for j in range(35):
            if loss < best_loss_main[j]:
                best_loss_main[j] = loss
                generators_main[j].save(f"/home/tk/Desktop/pensja/GEN_main_{j}.keras")
        for j in range(4):
            if loss < best_loss_bonus[j]:
                best_loss_bonus[j] = loss
                generators_bonus[j].save(f"/home/tk/Desktop/pensja/GEN_bonus_{j}.keras")

train(data, epochs=500)
