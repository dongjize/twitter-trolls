import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]

y_s = [0.5307, 0.5702, 0.5718, 0.6108, 0.6130]
y_m = [0.6466, 0.6295, 0.6326, 0.6326, 0.6540]
y_l = [0.7308, 0.6965, 0.6005, 0.6635, 0.7001]

group_labels = [
    'NB',
    'LR',
    'KNN',
    'DT',
    'RF'
]
plt.title('Accuracies of different ML algorithms\n')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')

plt.plot(x, y_s, 'r', label='small', marker='.')
plt.plot(x, y_m, 'b', label='medium', marker='.')
plt.plot(x, y_l, 'y', label='large', marker='.')
plt.xticks(x, group_labels, rotation=0)
plt.ylim(0, 1)

plt.legend(title="Dataset", bbox_to_anchor=[1, 0.3])
plt.grid()
plt.savefig("plot.png")
plt.show()
