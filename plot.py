import matplotlib.pyplot as plt
import statistics
import numpy as np

desktop = [0.10500001907348633, 0.03300046920776367, 0.030002355575561523, 0.027997970581054688, 0.03099989891052246,
           0.030000686645507812, 0.029999971389770508, 0.030997753143310547, 0.030001401901245117, 0.029999971389770508,
           0.030998945236206055, 0.029999732971191406, 0.028000593185424805, 0.028999805450439453, 0.028998851776123047,
           0.029001712799072266, 0.029999732971191406, 0.03200030326843262, 0.030001163482666016, 0.0279996395111084]
desktopLite = [0.02299952507019043, 0.01399993896484375, 0.013999700546264648, 0.01200103759765625,
               0.011999130249023438, 0.013999223709106445, 0.015000104904174805, 0.01300048828125, 0.011999845504760742,
               0.016002178192138672, 0.01599907875061035, 0.011999845504760742, 0.012998819351196289,
               0.013000011444091797, 0.011999368667602539, 0.011999845504760742, 0.01300048828125, 0.013000011444091797,
               0.013000249862670898, 0.011999130249023438]
piLite = [1.3107802867889404, 0.4541666507720947, 0.44028186798095703, 0.44707679748535156, 0.436492919921875,
          0.4487779140472412, 0.5254099369049072, 0.5145645141601562, 0.5063009262084961, 0.43390965461730957,
          0.44235754013061523, 0.4580402374267578, 0.4457542896270752, 0.4523308277130127, 0.42366862297058105,
          0.4308497905731201, 0.510239839553833, 0.4403398036956787, 0.46483302116394043, 0.44028782844543457]
pi = [3.515028238296509, 0.8430097103118896, 0.8305208683013916, 0.8398268222808838, 0.8601727485656738,
      0.8268671035766602, 0.8823099136352539, 0.8913218975067139, 0.8638620376586914, 0.8195717334747314,
      0.8260810375213623, 1.3288028240203857, 0.8375279903411865, 0.8354134559631348, 0.8388626575469971,
      0.8326609134674072, 0.8652079105377197, 0.8233113288879395, 0.8848562240600586, 0.8306427001953125]

min = np.array([min(desktop), min(desktopLite), min(piLite),min(pi)])
max = np.array([max(desktop), max(desktopLite), max(piLite), max(pi)])
mean = np.array([statistics.mean(desktop), statistics.mean(desktopLite), statistics.mean(piLite), statistics.mean(pi)])
std = np.array([statistics.stdev(desktop), statistics.stdev(desktopLite), statistics.stdev(piLite), statistics.stdev(pi)])
numberOfDecimals = 3
print('Desktop', round(std[0], numberOfDecimals), round(mean[0], numberOfDecimals), round(max[0], numberOfDecimals),
      round(min[0], numberOfDecimals))
print('Desktop LITE', round(std[1], numberOfDecimals), round(mean[1], numberOfDecimals), round(max[1], numberOfDecimals)
      , round(min[1], numberOfDecimals))
print('PI LITE', round(std[2], numberOfDecimals), round(mean[2], numberOfDecimals), round(max[2], numberOfDecimals),
      round(min[2], numberOfDecimals))
print('PI', round(std[3], numberOfDecimals), round(mean[3], numberOfDecimals), round(max[3], numberOfDecimals),
      round(min[3], numberOfDecimals))

desktopIncrease = (mean[1] - mean[0]) / mean[0]
piIncrease = (mean[2] - mean[3]) / mean[3]

print("Desktop Increase:", desktopIncrease, "Pi Increase:", piIncrease)


plt.errorbar(np.arange(4), mean, std, fmt='ok', lw=3)
plt.errorbar(np.arange(4), mean, [mean - min, max - mean],
             fmt='.k', ecolor='gray', lw=1)
plt.xlim(-1, 5)
