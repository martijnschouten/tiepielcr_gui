from telnetlib import X3PAD
from TiePieLCR_calibration import TiePieLCR_calibration
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# name = 'fe1_390pC_6.8p_30meg_calibration2_api'
# Ctest = 6.8e-12
# Rtest = 29.5e6

# name = 'fe1_390pC_6.8p_calibration_api'
# Ctest = 6.8e-12
# Rtest = 1e99

# name = 'fe1_HcurI_49R9_2k_calibration2_api'
# Ctest = 1e-18
# Rtest = 49.9

# name = 'fe3_390pC_6.8p_calibration_api'
# Ctest = 6.8e-12
# Rtest = 1e99

# name = 'fe4_68R_calibration2_api'
# Ctest = 0e-12
# Rtest = 67.708

# name = 'fe3_6p8pF_direct_AWG_connection3_api'
# Ctest = 6.8e-12
# Rtest = 1e99
# reference = 3
# gain = 0
# frontend = 3

# name = ['fe3_6p8pF_direct_AWG_connection3_api','fe3_22pF_direct_AWG_connection_api']
# Ctest = [6.8e-12,22e-12]
# Rtest = [1e99,1e99]
# reference = 3
# gain = 0
# frontend = 3

name = ['fe3_22pF_direct_AWG_connection_ground_plate_api']
Ctest = [22e-12]
Rtest = [1e99]
reference = 3
gain = 0
frontend = 3

# name = 'fe3_370uApV_10k_calibration_api'
# Ctest = 1.1e-12
# Rtest = 10015
# reference = 1
# gain = 0
# frontend = 3

# name = 'fe3_5uApV_1M2_calibration_api'
# Ctest = 1.3e-12
# Rtest = 1.175e6
# reference = 2
# gain = 0
# frontend = 3

# name = 'fe1_5uApV_1M2_calibration_api'
# Ctest = 1.3e-12
# Rtest = 1.175e6
# reference = 2
# gain = 0
# frontend = 1

# name = 'fe1_370uApV_10k_calibration_api'
# Ctest = 1.1e-12
# Rtest = 10015
# reference = 1
# gain = 0
# frontend = 1


deg = 15

min_er = 0.1

calibration = TiePieLCR_calibration()
calibration.load_calibration('calibrations/%s.yaml'%(frontend))

Ztest_list = []
use_list = []
f_list = []
Z_list = []
for i1 in range(len(name)):
    api_data = sio.loadmat('calibrations/data/'+name[i1]+'.mat')
    f = api_data['f'][0]
    X = api_data['X'][0]
    Y = api_data['Y'][0]
    X_e = api_data['error1'][0]
    Y_e = api_data['error2'][0]

    Z = X+1j*Y
    magn = np.absolute(Z)
    angle = np.angle(Z)

    X_er = X_e/magn
    Y_er = Y_e/magn
    use = np.where((X_er<min_er)&(Y_er<min_er))
    use_list.append(use)

    f_list.append(f)
    Z_list.append(Z)
    Ztest_list.append(Rtest[i1]/(1j*Rtest[i1]*f*2*np.pi*Ctest[i1]+1))

    f = f[use]
    X = X[use]
    Y = Y[use]
    Z = Z[use]
    magn = magn[use]
    angle = angle[use]

    Ztest = Rtest[i1]/(1j*Rtest[i1]*f*2*np.pi*Ctest[i1]+1)
    magn_test = np.absolute(Ztest)
    angle_test = np.angle(Ztest)
    a = np.real(Z)
    b = np.imag(Z)
    Rp = (b**2+a**2)/a
    Cp = -b/Rp/a/2/np.pi/f

    plt.figure()
    plt.subplot(2,1,1)
    plt.semilogx(f,X,f,np.real(Ztest))
    plt.ylabel('Real (Ohm)')
    plt.subplot(2,1,2)
    plt.loglog(f,-Y,f,-1*np.imag(Ztest))
    plt.ylabel('Imaginary (Ohm)')
    plt.xlabel('Frequency (Hz)')
    plt.legend(['Measured', 'Expected'])


    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(f,magn,f,magn_test)
    plt.ylabel('Magnitude (Ohm)')
    plt.subplot(2,1,2)
    plt.semilogx(f,angle,f,angle_test)
    plt.ylabel('Angle (rad)')
    plt.xlabel('Frequency (Hz)')
    plt.legend(['Measured', 'Expected'])

    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(f,Rp)
    plt.ylabel('Parallel resistance (Ohm)')
    plt.subplot(2,1,2)
    plt.semilogx(f,Cp)
    plt.ylabel('Angle (rad)')
    plt.xlabel('Parallel capacitance (Hz)')
    plt.legend(['Measured', 'Expected'])

use_in_all = use_list[0][0]            
for i1 in range(1,len(use_list)):
    use_in_all = np.intersect1d(use_in_all,use_list[i1])

f = f_list[0][use_in_all]
Z_mat = np.zeros((len(name),len(use_in_all)),dtype=complex)
Ztest_mat = np.zeros((len(name),len(use_in_all)),dtype=complex)

for i1 in range(len(name)):
    Z_mat[i1,:] = Z_list[i1][use_in_all]
    Ztest_mat[i1,:] = Ztest_list[i1][use_in_all]

calibration.set_poly_calibration(f,reference,gain,Z_mat,Ztest_mat,deg)
f_log = np.log10(f)
p_mag = np.poly1d(calibration.z_error_mag_rel[gain][reference])
p_ang = np.poly1d(calibration.z_error_angle[gain][reference])

calibration.save_calibration('calibrations/%s.yaml'%(frontend))

for i1 in range(len(name)):
    f = f_list[i1][use_list[i1]]
    f_log = np.log10(f)
    magn = np.absolute(calibration.get_compensated_z(f,reference,gain,Z_list[i1][use_list[i1]]))
    magn = magn-magn*p_mag(f_log)
    angle = np.angle(calibration.get_compensated_z(f,reference,gain,Z_list[i1][use_list[i1]]))
    magn_test = np.absolute(Ztest_list[i1][use_list[i1]])
    angle_test = np.angle(Ztest_list[i1][use_list[i1]])
    

    plt.figure()
    plt.subplot(2,1,1)
    plt.loglog(f,magn_test,f,magn)
    plt.ylabel('Relative error magnitude')
    plt.subplot(2,1,2)
    plt.semilogx(f,angle_test,f,angle-p_ang(f_log))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Error angle(rad)')
    plt.legend(['Measurement','Fit'])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.semilogx(f,p_mag(f_log)-(magn-magn_test)/magn)
    # plt.ylabel('Remaining error magnitude (%)')
    # plt.subplot(2,1,2)
    # plt.semilogx(f,angle-angle_test-p_ang(f_log))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Remaining error angle(rad)')


plt.show()