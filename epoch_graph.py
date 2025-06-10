import matplotlib.pyplot as plt

# Epochs
epochs = [20, 40, 60, 80, 100]

# Accuracy values for each scenario
acc_scenario1 = [0.668, 0.810, 0.994, 0.995, 0.996]  # Deep, all params
acc_scenario2 = [0.668, 0.668, 0.668, 0.668, 0.668]  # Deep, 6 params
acc_scenario3 = [0.332, 0.572, 0.572, 0.572, 0.572]  # Shallow, 6 params
acc_scenario4 = [0.297, 0.478, 0.619, 0.752, 0.811]  # Shallow, all params

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, acc_scenario1, label="Deep, All Params", marker='o')
plt.plot(epochs, acc_scenario2, label="Deep, 6 Params", marker='o')
plt.plot(epochs, acc_scenario3, label="Shallow, 6 Params", marker='o')
plt.plot(epochs, acc_scenario4, label="Shallow, All Params", marker='o')

plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.ylim(0.25, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
