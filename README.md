2D Neural Network Classifier (Backprop + GUI)

A from-scratch neural network implementation for classifying 2D data points, enhanced with a graphical interface for experimenting with training scenarios. The project includes manual backpropagation, multiple optimizers (Adam, Momentum, SGD), data generation controls, and real-time visualization of decision boundaries.

⸻

🚀 Features
	•	Neural network from scratch (NumPy)
	•	Manual backpropagation
	•	Multiple optimizers:
	•	Adam
	•	Momentum
	•	Standard Gradient Descent
	•	Interactive GUI (Tkinter):
	•	Visualize training in real time
	•	Adjust dataset size and noise
	•	Explore different training scenarios
	•	Compare optimizer behaviors
	•	2D data classification with decision boundary visualization
	•	Cross-platform support (Windows / macOS / Linux)

⸻

🖥️ Usage

Run the GUI instead of the example script:

python gui.py

The GUI allows you to:
	•	Set number of data points
	•	Add noise to the dataset
	•	Select optimizer
	•	Tune learning rate and other hyperparameters
	•	Start/stop training live
	•	Watch the decision boundary evolve in real time

⸻

📦 Requirements
	•	Python 3.8+
	•	Tkinter (included in most Python installations)
	•	NumPy
	•	Matplotlib (if used for visualization)

⸻

🍏 macOS / Linux Note

On macOS or Linux, running the project inside a Conda environment is recommended, especially for Tkinter compatibility:

conda create -n nn2d python=3.10
conda activate nn2d


⸻

📁 Project Structure

├── gui.py              # Main GUI interface
├── example.py          # Old example (use gui.py instead)
├── nn/                 # Neural network implementation
├── optimizers/         # Adam, Momentum, SGD implementations
├── utils/              # Dataset generation, plotting, helpers
├── requirements.txt
└── README.md


⸻

🎯 Project Purpose

This project is designed for experimentation and learning. It provides insight into how neural networks train by allowing you to:
	•	See gradients and decision boundaries update live
	•	Compare optimizers like Adam and Momentum
	•	Interactively control dataset complexity
	•	Understand the impact of noise and sample size

