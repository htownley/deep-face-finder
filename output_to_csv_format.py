import os


print "Epoch: " + "\t" + "test accuracy " + "\t" + "train accuracy "

with open("simple_cnn_output.txt") as f:
    content = f.readlines()

    for i in range(len(content)):
    	if i % 3 == 0:
    		my_out = content[i][len("Epoch: "):].strip() + "\t" + content[i+2][len("test accuracy "):].strip() + "\t" + content[i+1][len("train accuracy "):].strip()
    		print my_out








