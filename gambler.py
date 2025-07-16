import tensorflow as tf
import numpy as np
import os, sys
with tf.device('CPU:0'):
    # Load and preprocess dataset
    data = np.loadtxt("lotto.txt", dtype=np.float32)
    data = np.round(data)
    data = data[np.newaxis, :, :]  # Shape: (1,1, 6)
    data = tf.convert_to_tensor(data)

    def generator_loss(pred1, inp1):
        no = 0
        diff1 = 0.0
        diff2 = 0.0
        pred_vals = tf.reshape(pred1, (6,))
        target_vals = tf.reshape(inp1, (6,))

        for j in range(6):
            diff = pred_vals[j] - target_vals[j]
            if (pred_vals[j] >= target_vals[j] - 0.2) and (pred_vals[j] <= target_vals[j] + 0.2):
                no += 1
                diff2 += 1e-6
            else:
                diff2 += diff ** 2
            if target_vals[j] < 0.5:
                diff2 += 10 * diff ** 2

        if no == 3:
            diff1 -= 9e2 * no
        if no == 4:
            diff1 -= 9e4 * no
        if no == 5:
            diff1 -= 9e8 * no
        if no == 6:
            diff1 -= 9e18 * no
        return diff1 + diff2

    def create_generator_model(index):
        input_tensor = tf.keras.Input(shape=(200, 6), batch_size=1, name=f"input_{index}")
        x = tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu")(input_tensor)
        x = tf.keras.layers.Conv1D(32, 1, padding="causal", activation="linear")(x)
        # x = tf.keras.layers.Dense(32, activation="gelu")(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(6, activation="softplus")(x)
        return tf.keras.Model(inputs=input_tensor, outputs=x, name=f"Generator_{index}")


    generators = [create_generator_model(j) for j in range(49)]
    optimizers = [tf.keras.optimizers.AdamW(1e-4) for _ in range(49)]

    def train_step(model, optimizer, inp, target):
        with tf.GradientTape() as tape:
            generated = model(inp, training=True)
            g_loss = generator_loss(generated, target)
        grads = tape.gradient(g_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return g_loss, generated


    def train(dataset, epochs):
        best_loss = [float('inf')] * 49
        total_iters = len(dataset[0]) - 1

        for epoch in range(epochs):
            total_loss = [0.0] * 49
            progress_line = ""
            # for window in range(len(data)-201):
            for i in range(total_iters):
                inp = data[:, i:i + 200, :]            # shape: (1, 200, 6)
                target = data[:, i + 200:i + 201, :]   # shape: (1, 1, 6)


                active_branches = set()

                for feature_index in range(6):
                    j = int(inp[:, 0, feature_index])
                    j =int(j-1)  # clamp safely
                    model = generators[j]
                    optimizer = optimizers[j]

                    loss, outt = train_step(model, optimizer, inp, target)
                    total_loss[j] += loss
                    active_branches.add(j)

                percent = int((i + 1) / total_iters * 100)
                active_str = ', '.join(str(idx) for idx in sorted(active_branches))
                progress_line = (
                    f"\rEpoch {epoch+1}/{epochs} | {percent}% | Step {i+1}/{total_iters} | "
                    f"Branches updated: [{active_str}] | Last loss: {loss:.4f} | out: {outt}"
                )
                sys.stdout.write(progress_line)
                sys.stdout.flush()

            sys.stdout.write("\r" + " " * len(progress_line) + "\r")
            print(f"✅ Epoch {epoch + 1} complete")

            # Save models with best loss
            for j in range(49):
                if total_loss[j] < best_loss[j]:
                    best_loss[j] = total_loss[j]
                    path = f"/home/tk/Desktop/lotto/GENERATOR_branch_{j}.keras"
                    generators[j].save(path)
                    print(f"💾 Model saved for branch {j} at epoch {epoch + 1} with loss {best_loss[j]:.4f}")

    
    train(data, 150)
