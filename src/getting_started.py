# %%
from spektral.datasets import citation

A, X, y, train_mask, val_mask, test_mask = citation.load_data("cora")

N = A.shape[0]
F = X.shape[-1]
n_classes = y.shape[-1]

# %%
X[1, 2]
# %%
from spektral.layers import GraphConv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout


# %%
X_in = Input(shape=(F,))
A_in = Input((N,), sparse=True)

X_1 = GraphConv(16, "relu")([X_in, A_in])
X_1 = Dropout(0.5)(X_1)
X_2 = GraphConv(n_classes, "softmax")([X_1, A_in])

model = Model(inputs=[X_in, A_in], outputs=X_2)


# %%
A = GraphConv.preprocess(A).astype("f4")


# %%
model.compile(
    optimizer="adam", loss="categorical_crossentropy", weighted_metrics=["acc"]
)
model.summary()


# %%
# Prepare data
X = X.toarray()
A = A.astype("f4")
validation_data = ([X, A], y, val_mask)

# Train model
model.fit(
    [X, A],
    y,
    sample_weight=train_mask,
    validation_data=validation_data,
    batch_size=N,
    shuffle=False,
)


# %%
# Evaluate model
eval_results = model.evaluate([X, A], y, sample_weight=test_mask, batch_size=N)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))
