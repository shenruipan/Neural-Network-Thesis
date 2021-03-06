import matplotlib
import matplotlib.pylab as plt
import numpy as np

text_file = open("C:/Users/Veblen/mystuff/enlarge_net_data_1000/test_accuracy.txt", "r")
PanNet_enlarged_1 = np.asarray(text_file.readlines())
text_file = open("C:/Users/Veblen/mystuff/enlarge_net_data_1000/LeNet1-enlarge_test_accuracy.txt", "r")
LeNet1_enlarged_1 = np.asarray(text_file.readlines())
text_file = open("C:/Users/Veblen/mystuff/enlarge_net_data_1000/backPanNet-enlarge_test_accuracy.txt", "r")
backPanNet_enlarged_1 = np.asarray(text_file.readlines())
text_file = open("C:/Users/Veblen/mystuff/enlarge_net_data/test_accuracy.txt", "r")
PanNet_enlarged_2 = np.asarray(text_file.readlines())
text_file = open("C:/Users/Veblen/mystuff/enlarge_net_data/LeNet1-enlarge_test_accuracy.txt", "r")
LeNet1_enlarged_2 = np.asarray(text_file.readlines())
text_file = open("C:/Users/Veblen/mystuff/enlarge_net_data/backPanNet-enlarge_test_accuracy.txt", "r")
backPanNet_enlarged_2 = np.asarray(text_file.readlines())

PanNet_enlarged = np.concatenate((PanNet_enlarged_1, PanNet_enlarged_2), axis = 0)
LeNet1_enlarged = np.concatenate((LeNet1_enlarged_1, LeNet1_enlarged_2), axis = 0)
backPanNet_enlarged = np.concatenate((backPanNet_enlarged_1, backPanNet_enlarged_2), axis = 0)

num_training1=[(k+1)*10*100 for k in range(int(1000/10))]
num_training2=[(k+1)*1000*100 for k in range(int(50000/1000))]
num_training =num_training1+num_training2

plt.figure(1)
#plt.axis((-1000,50000,0.00,1.00))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
PanNet_enlarged_plot,=plt.plot(num_training, PanNet_enlarged, label='PanNet-enlarged')
LeNet1_enlarged_plot,=plt.plot(num_training, LeNet1_enlarged, label='LeNet-1-enlarged')
backPanNet_enlarged_plot,=plt.plot(num_training, backPanNet_enlarged, label='BackPanNet-enlarged')
plt.xlabel('Number of training images')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend(handles=[PanNet_enlarged_plot, LeNet1_enlarged_plot, backPanNet_enlarged_plot])
plt.savefig('enlarged_net_accu_plot_combined.png', bbox_inches='tight')

plt.figure(2)
#plt.axis((0,1000,0.00,1.00))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
PanNet_enlarged_plot_1,=plt.plot(num_training1, PanNet_enlarged_1, label='PanNet-enlarged')
LeNet1_enlarged_plot_1,=plt.plot(num_training1, LeNet1_enlarged_1, label='LeNet-1-enlarged')
backPanNet_enlarged_plot_1,=plt.plot(num_training1, backPanNet_enlarged_1, label='BackPanNet-enlarged')
plt.xlabel('Number of training images')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend(handles=[PanNet_enlarged_plot_1, LeNet1_enlarged_plot_1, backPanNet_enlarged_plot_1])
plt.savefig('enlarged_net_accu_plot_combined_1.png', bbox_inches='tight')

plt.figure(3)
plt.axis((0,52000*100,0.90,1.00))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
PanNet_enlarged_plot_2,=plt.plot(num_training2, PanNet_enlarged_2, label='PanNet-enlarged')
LeNet1_enlarged_plot_2,=plt.plot(num_training2, LeNet1_enlarged_2, label='LeNet-1-enlarged')
backPanNet_enlarged_plot_2,=plt.plot(num_training2, backPanNet_enlarged_2, label='BackPanNet-enlarged')
plt.xlabel('Number of training images')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend(handles=[PanNet_enlarged_plot_2, LeNet1_enlarged_plot_2, backPanNet_enlarged_plot_2])
plt.savefig('enlarged_net_accu_plot_combined_2.png', bbox_inches='tight')


plt.figure(4)
fig1, ax1 = plt.subplots()
PanNet_enlarged_plot_3,=plt.plot(np.asarray(num_training), np.asarray([(float(i)) for i in PanNet_enlarged]), label='PanNet-enlarged')
LeNet1_enlarged_plot_3,=plt.plot(np.asarray(num_training), np.asarray([(float(i)) for i in LeNet1_enlarged]), label='LeNet-1-enlarged')
backPanNet_enlarged_plot_3,=plt.plot(np.asarray(num_training), np.asarray([(float(i)) for i in backPanNet_enlarged]), label='BackPanNet-enlarged')
ax1.set_xscale('log')
ax1.set_xticks([1e3, 1e5, 1e6, 5e6])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
plt.xlabel('Number of training images')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend(handles=[PanNet_enlarged_plot_3, LeNet1_enlarged_plot_3, backPanNet_enlarged_plot_3])
plt.savefig('enlarged_net_accu_plot_combined_3.png', bbox_inches='tight')


