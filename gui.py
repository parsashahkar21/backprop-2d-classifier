# gui.py
# Interactive trainer for your tiny-autograd MLP with live decision boundary.
# - Optimizer picker (SGD / SGD with momentum / Adam)
# - Hyperparams with "auto if 0"
# - Dataset complexity + epochs
# - Live Matplotlib plot inside Tkinter
#
# Run: python gui.py

# ---- suppress macOS deprecated Tk warning (must be BEFORE tkinter import) ----
import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"

import threading
import queue
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----- import your engine pieces -----
from tensor import Tensor
from function import ReLU, CrossEntropyWithSoftmax
from module import Module, Linear
from optimizer import SGD, Adam


# ---------- Data ----------
def create_spiral_data(points_per_class, num_classes, noise=0.2, seed=0):
    rng = np.random.RandomState(seed)
    X = np.zeros((points_per_class * num_classes, 2), dtype=np.float32)
    y = np.zeros(points_per_class * num_classes, dtype='uint8')
    for c in range(num_classes):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1, points_per_class)
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.randn(points_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


# ---------- Model ----------
class MLP(Module):
    def __init__(self, num_features, num_classes, hidden=100):
        super().__init__()
        self.fc1 = Linear(num_features, hidden)
        self.fc2 = Linear(hidden, num_classes)

    def forward(self, X: Tensor):
        h = ReLU.apply(self.fc1(X))
        out = self.fc2(h)
        return out

    def predict(self, X):
        return self.forward(X).data


# ---------- Visualization helpers ----------
def compute_grid_pred(model, X, h=0.02):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.argmax(model.predict(Tensor(grid, requires_grad=False)), axis=1)
    Z = Z.reshape(xx.shape)
    return xx, yy, Z


def draw_decision_boundary(ax, model, X, y):
    ax.clear()
    xx, yy, Z = compute_grid_pred(model, X, h=0.02)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.35)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Spectral, edgecolors='k')
    ax.set_title("Decision Boundary (live)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


# ---------- Hyperparameter auto-pick ("auto if 0") ----------
def auto_hparams(optimizer_name, lr, momentum, beta1, beta2, eps):
    if optimizer_name == "SGD":
        if lr == 0.0: lr = 0.1
        if momentum == 0.0: momentum = 0.0
    elif optimizer_name == "SGD (momentum)":
        if lr == 0.0: lr = 0.05
        if momentum == 0.0: momentum = 0.9
    elif optimizer_name == "Adam":
        if lr == 0.0: lr = 1e-3
        if beta1 == 0.0: beta1 = 0.9
        if beta2 == 0.0: beta2 = 0.999
        if eps == 0.0: eps = 1e-8
    return lr, momentum, beta1, beta2, eps


# ---------- Training worker ----------
def train_worker(
    q,
    optimizer_name,
    lr, momentum, beta1, beta2, eps,
    points_per_class, num_classes, noise,
    hidden, epochs, seed
):
    lr, momentum, beta1, beta2, eps = auto_hparams(optimizer_name, lr, momentum, beta1, beta2, eps)

    X, y = create_spiral_data(points_per_class, num_classes, noise=noise, seed=seed)
    X_tensor = Tensor(X)
    num_samples = len(y)
    y_one_hot = np.zeros((num_samples, num_classes), dtype=np.float32)
    y_one_hot[np.arange(num_samples), y] = 1.0
    y_true = Tensor(y_one_hot, requires_grad=False)

    model = MLP(num_features=2, num_classes=num_classes, hidden=hidden)

    if optimizer_name == "SGD":
        optimizer = SGD(model.parameters(), lr=lr)
    elif optimizer_name == "SGD (momentum)":
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "Adam":
        optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    else:
        raise ValueError("Unknown optimizer")

    report_every = max(epochs // 50, 1)  # ~50 UI updates
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        y_pred = model.forward(X_tensor)
        loss = CrossEntropyWithSoftmax.apply(y_pred, y_true)
        loss.backward()
        optimizer.step()

        if epoch % report_every == 0:
            q.put({
                "epoch": epoch,
                "loss": float(loss.data),
                "model_params": [p.data.copy() for p in model.parameters()],
                "X": X, "y": y
            })

    q.put({
        "epoch": epochs,
        "loss": float(loss.data),
        "model_params": [p.data.copy() for p in model.parameters()],
        "X": X, "y": y,
        "done": True
    })


# ---------- Tkinter App ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MLP Trainer — pick optimizer & hyperparams")
        self.geometry("1100x700")

        self.q = queue.Queue()
        self.worker_thread = None

        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(ctrl, text="Optimizer").grid(row=0, column=0, sticky="w")
        self.opt_choice = ttk.Combobox(ctrl, values=["SGD", "SGD (momentum)", "Adam"], state="readonly")
        self.opt_choice.current(0)
        self.opt_choice.grid(row=0, column=1, sticky="ew", pady=2)

        self.lr_var = tk.DoubleVar(value=0.1)
        self.mom_var = tk.DoubleVar(value=0.0)
        self.b1_var = tk.DoubleVar(value=0.9)
        self.b2_var = tk.DoubleVar(value=0.999)
        self.eps_var = tk.DoubleVar(value=1e-8)

        def add_entry(label, var, r, hint="(0 = auto)"):
            ttk.Label(ctrl, text=label).grid(row=r, column=0, sticky="w")
            e = ttk.Entry(ctrl, textvariable=var, width=12)
            e.grid(row=r, column=1, sticky="w", pady=2)
            ttk.Label(ctrl, text=hint, foreground="#666").grid(row=r, column=2, sticky="w")
            return e

        add_entry("Learning rate", self.lr_var, 1)
        add_entry("Momentum", self.mom_var, 2)
        add_entry("Adam β1", self.b1_var, 3)
        add_entry("Adam β2", self.b2_var, 4)
        add_entry("Adam ε", self.eps_var, 5)

        # Dataset complexity
        ttk.Separator(ctrl).grid(row=6, columnspan=3, sticky="ew", pady=8)
        ttk.Label(ctrl, text="Dataset / Model / Training").grid(row=7, column=0, sticky="w")

        self.ppc_var = tk.IntVar(value=100)
        self.cls_var = tk.IntVar(value=3)
        self.noise_var = tk.DoubleVar(value=0.2)
        self.hidden_var = tk.IntVar(value=100)
        self.epochs_var = tk.IntVar(value=5000)
        self.seed_var = tk.IntVar(value=0)

        def add_spin(label, var, r, frm=1, to=100000, inc=1):
            ttk.Label(ctrl, text=label).grid(row=r, column=0, sticky="w")
            sp = ttk.Spinbox(ctrl, from_=frm, to=to, increment=inc, textvariable=var, width=12)
            sp.grid(row=r, column=1, sticky="w", pady=2)
            return sp

        add_spin("Points/class", self.ppc_var, 8, frm=50, to=1000, inc=10)
        add_spin("Classes", self.cls_var, 9, frm=2, to=6, inc=1)
        add_spin("Hidden size", self.hidden_var, 10, frm=10, to=512, inc=10)
        add_spin("Epochs", self.epochs_var, 11, frm=500, to=50000, inc=500)
        add_spin("Seed", self.seed_var, 12, frm=0, to=9999, inc=1)

        ttk.Label(ctrl, text="Noise (0.0–0.8)").grid(row=13, column=0, sticky="w")
        noise_scale = ttk.Scale(ctrl, from_=0.0, to=0.8, variable=self.noise_var, orient=tk.HORIZONTAL, length=160)
        noise_scale.grid(row=13, column=1, columnspan=2, sticky="ew", pady=2)

        ttk.Separator(ctrl).grid(row=14, columnspan=3, sticky="ew", pady=8)
        self.btn_run = ttk.Button(ctrl, text="Train & Visualize", command=self.on_run)
        self.btn_run.grid(row=15, column=0, columnspan=2, sticky="ew", pady=4)

        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.btn_stop.grid(row=15, column=2, sticky="ew", pady=4)

        ttk.Label(ctrl, text="Log").grid(row=16, column=0, sticky="w")
        self.log = tk.Text(ctrl, width=40, height=12)
        self.log.grid(row=17, column=0, columnspan=3, pady=4, sticky="nsew")
        ctrl.grid_columnconfigure(1, weight=1)

        fig = plt.Figure(figsize=(7.5, 6), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Click 'Train & Visualize' to start")
        self.ax.set_xlabel("x1")
        self.ax.set_ylabel("x2")
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas.draw()  # initial draw

        self.after(100, self.poll_queue)

    def on_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return

        optimizer_name = self.opt_choice.get()
        lr = float(self.lr_var.get())
        momentum = float(self.mom_var.get())
        b1 = float(self.b1_var.get())
        b2 = float(self.b2_var.get())
        eps = float(self.eps_var.get())

        ppc = int(self.ppc_var.get())
        classes = int(self.cls_var.get())
        noise = float(self.noise_var.get())
        hidden = int(self.hidden_var.get())
        epochs = int(self.epochs_var.get())
        seed = int(self.seed_var.get())

        X0, y0 = create_spiral_data(ppc, classes, noise=noise, seed=seed)
        self.ax.clear()
        self.ax.scatter(X0[:, 0], X0[:, 1], c=y0, s=20, cmap=plt.cm.Spectral, edgecolors="k")
        self.ax.set_title("Dataset preview (training about to start)")
        self.ax.set_xlabel("x1"); self.ax.set_ylabel("x2")
        self.canvas.draw_idle()

        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log_insert(f"Starting: {optimizer_name}, lr={lr}, momentum={momentum}, "
                        f"β1={b1}, β2={b2}, ε={eps}\n"
                        f"Data: ppc={ppc}, classes={classes}, noise={noise}, hidden={hidden}, epochs={epochs}\n"
                        f"(Any zeros will be auto-filled)\n")

        self.q = queue.Queue()  # reset queue for a new run
        self.worker_thread = threading.Thread(
            target=train_worker,
            args=(self.q, optimizer_name, lr, momentum, b1, b2, eps,
                  ppc, classes, noise, hidden, epochs, seed),
            daemon=True
        )
        self.worker_thread.start()

    def on_stop(self):
        self.q = queue.Queue()
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_run.config(state=tk.NORMAL)
        self.log_insert("Stopped (cooperative). You can start a new run.\n")

    def poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                epoch = msg["epoch"]
                loss = msg["loss"]
                X, y = msg["X"], msg["y"]

                # Rebuild a temp model using parameter shapes to avoid mismatches
                W1, b1, W2, b2 = msg["model_params"]
                in_features, hidden = W1.shape
                hidden2, num_classes = W2.shape
                if hidden != hidden2:
                    raise RuntimeError("Hidden sizes disagree between layers.")
                model = MLP(num_features=in_features, num_classes=num_classes, hidden=hidden)

                params = model.parameters()
                # Strictly assign in the known order: [W1, b1, W2, b2]
                params[0].data = W1
                params[1].data = b1
                params[2].data = W2
                params[3].data = b2

                # update plot
                draw_decision_boundary(self.ax, model, X, y)
                self.canvas.draw_idle()

                self.log_insert(f"Epoch {epoch} | Loss {loss:.4f}\n")

                if msg.get("done", False):
                    self.btn_stop.config(state=tk.DISABLED)
                    self.btn_run.config(state=tk.NORMAL)
                    self.log_insert("Training finished.\n")
        except queue.Empty:
            pass
        # schedule next poll
        self.after(100, self.poll_queue)

    def log_insert(self, text):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)


if __name__ == "__main__":
    App().mainloop()