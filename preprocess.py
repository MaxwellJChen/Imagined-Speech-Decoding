from DataLoader import norm, norm_calc, printer

import numpy as np

all = np.load('raw-10.npy')
ranges = np.concatenate([range(30), range(300, 330), range(600, 630)])
x_ica1_train = np.delete(all, ranges, axis = 0)
printer(x_ica1_train)
x_ica1_test = all[ranges][:][:]
y_ica1_train = np.zeros(810)
y_ica1_train[270:540] = 1
y_ica1_train[540:] = 2
y_ica1_test = np.zeros(90)
y_ica1_test[30:60] = 1
y_ica1_test[60:] = 2

x_ica1_train = norm(x_ica1_train)
x_ica1_test = norm(x_ica1_test)


np.save('x_ica1_train', x_ica1_train)
np.save('x_ica1_test', x_ica1_test)
np.save('y_ica1_train', y_ica1_train)
np.save('y_ica1_test', y_ica1_test)
printer(ranges)
printer(x_ica1_train)
printer(x_ica1_test)
# printer(all)