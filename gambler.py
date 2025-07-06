import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

opti = tf.keras.optimizers.AdamW(1e-4)

data = np.loadtxt("lotto.txt", dtype=np.float32)
data = np.round(data)

# reshape to (T, 1, 6) — treating each row as a sample with shape (1, 6)
data = data[np.newaxis, :,  :]  # shape: (T, 1, 6)
data = tf.convert_to_tensor(data)


def generator_loss(pred1,inp1):
    no=0
    diff1=0.0
    diff2=0.0
    # no2=1
    # print(inp1[0])
    # for i in range(len(inp1[0])):
    for j in range(len(inp1[0,0])):
        if inp1[0,0,j]-0.2<=pred1[0,0,j]<=inp1[0,0,j]+0.2:     
            
            no+=1 
            diff2-=1e2
            
        else:
            diff2+=(pred1[0,0,j]-inp1[0,0,j])**6
    if no==3:
        diff1-=9e2*no
    if no==4:
        diff1-=9e4*no
    if no==5:
        diff1-=9e8*no
    if no==6:
        diff1-=9e18*no
    diff2+=(6-no)**8
    return diff1+diff2




# Function to create each individual branch block
def create_branch_block(input_tensor, i, j, n, dilation_rate):
    x = tf.keras.layers.Conv1D(4*n, kernel_size=3, padding="causal",
                            activation="gelu", dilation_rate=dilation_rate,
                            name=f'branch_i{i}_j{j}_1')(input_tensor)
    x = tf.keras.layers.Conv1D(n, kernel_size=3, padding="causal",
                            activation="gelu", dilation_rate=dilation_rate//2,
                            name=f'branch_i{i}_j{j}_2')(x)
    x = tf.keras.layers.Dense(n*4, activation="gelu", name=f'branch_i{i}_j{j}_3')(x)
    x = tf.keras.layers.Dense(n, activation="gelu", name=f'branch_i{i}_j{j}_4')(x)
    return x

# Build the full model with all branches
def build_static_model():
    input_tensor = tf.keras.Input(shape=(1, 6),batch_size=1)  # shape: (batch, 1, 1, 6)
    squeezed = tf.keras.layers.Reshape((6,), name='squeeze')(input_tensor)  # shape: (batch, 6)

    all_outputs = []

    for i in range(6):  # Each feature position
        input_i = tf.keras.layers.Lambda(lambda x: x[:, i], name=f'feature_{i}')(squeezed)

        branch_outputs = []
        for j in range(49):  # Value-based routing
            condition = tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.equal(tf.cast(x, tf.int32), j), tf.float32),
                name=f'mask_i{i}_j{j}'
            )(input_i)  # shape: (batch,)

            # Expand dims to match Conv1D input shape
            branch_input = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, -1), -1),
                                                name=f'expand_i{i}_j{j}')(input_i)
            condition_exp = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, -1), -1),
                                                name=f'expand_mask_i{i}_j{j}')(condition)

            # Create branch
            branch_output = create_branch_block(branch_input, i, j, n=16, dilation_rate=24)  # shape: (batch, 1, units)
            masked_output = tf.keras.layers.Multiply(name=f'masked_i{i}_j{j}')([branch_output, condition_exp])
            branch_outputs.append(masked_output)

        # Sum all masked outputs for this feature position
        summed = tf.keras.layers.Add(name=f'sum_i{i}')(branch_outputs)
        all_outputs.append(summed)

    # Concatenate outputs from all 6 positions
    final_concat = tf.keras.layers.Concatenate(axis=-1, name='concatenate')(all_outputs)  # shape: (batch, 1, units * 6)
    final = tf.keras.layers.Dense(6, activation='sigmoid', name='final_dense')(final_concat)

    return tf.keras.Model(inputs=input_tensor, outputs=final, name='FullConditionalRoutingModel')

model = build_static_model()
model.compile()
model.summary()

def train_step(inp, comparable):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        outs = model(inp, training=True)
        # gen_loss = 
        gen_mse=generator_loss(comparable,outs)

    gradients_gen = gen_tape.gradient(gen_mse, model.trainable_variables)
    opti.apply_gradients(zip(gradients_gen, model.trainable_variables))
    # gen_loss_float = float(gen_mse.numpy())
    return gen_mse.numpy()

def train(dataset, epochs):
    best_loss = float('inf')  # Initialize with infinity for comparison

    for epoch in range(epochs):
        
        epoch_gen_loss = 0.0
        # fake_mse=0.0
        
        for i in tqdm(range(len(dataset[0])-1)):
            
            inp = data[:,i:i+1]         # shape (1, 1, 6)
            comparable = data[:,i+1:i+2]  # shape (1, 1, 6)
            gen_loss = train_step(inp, comparable)
            # fake_mse+=fake_acc
            
        # print(f"Epoch {epoch + 1}, loss= {epoch_gen_loss}")
        # model.save("best_gen.keras")
        
        if epoch_gen_loss<best_loss:
            best_loss = epoch_gen_loss
            model.save("GENERATOR_lotto.keras")
            print(f"Discriminator model saved with loss {best_loss}")
with tf.device('CPU:0'):
    train(data,50000*len(data[0]))
