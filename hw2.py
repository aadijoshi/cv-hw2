import math
import cmath
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.misc import imread
from scipy.signal import convolve2d

#define constants
Nx = 6
Ny = 6
d = 5

theta_array = [0+0j, cmath.pi/4+0j, cmath.pi/2+0j, 3*cmath.pi/4+0j]
sigma_array = [2+0j]
images_to_convolve = ["left.png", "right.png"]

theta_string_array = ["0", "Π/4","Π/2", "3Π/4"]

#figures for displaying plots		
fig1 = plt.figure(1)
fig1.canvas.set_window_title("Real Morlet")
fig2 = plt.figure(2)
fig2.canvas.set_window_title("Imaginary Morlet")

fig3 = plt.figure(3)
fig3.canvas.set_window_title("Convolved Real")
fig4 = plt.figure(4)
fig4.canvas.set_window_title("Convolved Imaginary")

fig5 = plt.figure(5)
fig5.canvas.set_window_title("Wavelet Edge Transformation")

fig6 = plt.figure(6)
fig6.canvas.set_window_title("Disparity Values")

fig6 = plt.figure(7)
fig6.canvas.set_window_title("Disparity Map")

all_images_real_convolved_arrays = []
all_images_imaginary_convolved_arrays = []

#define helper functions
def dotProduct(A, B):
	total = 0
	for i in range(0, len(A)):
		total = total + A[i]*B[i]
	return total

for im_string in images_to_convolve:
	real_convolved_arrays = []
	imaginary_convolved_arrays = []
	for s_n in range(0, len(sigma_array)):
		sigma = sigma_array[s_n]
		for t_n in range(0, len(theta_array)):
			theta = theta_array[t_n]
			e_theta = [cmath.cos(theta), cmath.sin(theta)]

			def morletWavelet(c1, c2, u):
				#outside = C1/σ
				outside = (c1/sigma)
				#leftAngle = (∏ / (2σ) )*(u.e_θ)
				leftAngle = (cmath.pi/(2*sigma))*(dotProduct(u,e_theta))
				leftSide = cmath.cos(leftAngle)+(1j*cmath.sin(leftAngle))
				#leftSide = cmath.exp(leftAngle*1j)
				inside = leftSide-c2
				#rightSide = e^ (-(u^2)/ ( 2 (σ^2) ) ) 
				rightSide = cmath.exp((-(dotProduct(u, u)))/(2*(sigma**2)))
				return (outside*inside*rightSide)

			#CALCULATE C2
			c2 = 0
			denominator = 0
			
			for x in range(-Nx, Nx+1):
				for y in range(-Ny, Ny+1):
					#u = (x, y)
					u = [x, y]
					#leftAngle = ( Π / (2σ) ) * (u.e_θ)
					leftAngle = (cmath.pi/(2*sigma))*(dotProduct(u,e_theta))
					
					leftSide = cmath.cos(leftAngle)+(1j*cmath.sin(leftAngle))
					
					#leftSide = e^(i * leftAngle)
					#leftSide = cmath.exp(leftAngle*1j)
					
					#rightSide = e^(-(u^2)/(2σ^2))
					rightSide = cmath.exp((-(dotProduct(u, u)))/(2*(sigma**2)))
					
					c2 = c2+(leftSide*rightSide)
					denominator = denominator+rightSide

			c2 = c2/denominator

			#CALCULATE C1
			Z = 0
			for x in range(-Nx, Nx+1):
				for y in range(-Ny, Ny+1):
					#u = (x,y)
					u = [x, y]

					#outside = e^(-(u^2) / (σ^2) )
					outside = cmath.exp((-(dotProduct(u, u)))/(sigma**2))

					#inside = 1+(c2^2)-(2*c2*cos( ( Π / (2σ) ) * (u.e_θ) ) )				
					#inside= 1+(c2**2)-(2*c2*cmath.cos((cmath.pi/(2*sigma))*dotProduct(u, e_theta)))

					leftAngle = (cmath.pi/(2*sigma))*(dotProduct(u,e_theta))

					left_e = cmath.cos(leftAngle)-(1j*cmath.sin(leftAngle))
					right_e = cmath.cos(leftAngle)+(1j*cmath.sin(leftAngle))

					#left_e = cmath.exp(leftAngle*(-1*1j))
					#right_e = cmath.exp(leftAngle*1j)

					inside = 1+(c2**2)-(c2*(left_e+right_e))

					Z = Z+(inside*outside)


			Z = (1/(sigma**2))*Z
			c1 = 1/cmath.sqrt(Z)

			#Create and display image
			real_values = np.zeros((2*Nx+1, 2*Ny+1))
			imaginary_values = np.zeros((2*Nx+1, 2*Ny+1))

			for x in range(-Nx, Nx+1):
				for y in range(-Ny, Ny+1):
					value = (morletWavelet(c1, c2, [x,y])).real
					real_values[x+Nx][y+Ny] = value

			for x in range(-Nx, Nx+1):
				for y in range(-Ny, Ny+1):
					value = (morletWavelet(c1, c2, [x,y])).imag
					imaginary_values[x+Nx][y+Ny] = value

			#Plot image
			plt.figure(1)
			ax = plt.subplot(3,4,s_n*4+t_n+1)
			ax.set_title('σ:'+str(sigma)+' θ:'+theta_string_array[t_n])
			plt.imshow(real_values, cmap="gray")
			plt.tight_layout()
			
			plt.figure(2)
			ax = plt.subplot(3,4,s_n*4+t_n+1)
			ax.set_title('σ:'+str(sigma)+' θ:'+theta_string_array[t_n])
			plt.imshow(imaginary_values, cmap="gray")
			plt.tight_layout()

			morlet_values = np.zeros(((2*Nx+1), (2*Ny+1)), dtype=complex)
			for x in range(-Nx, Nx+1):
				for y in range(-Ny, Ny+1):
					morlet_values[x+Nx][y+Ny] = (morletWavelet(c1, c2, [x,y]))

			assert np.abs(np.sum(np.abs(morlet_values) ** 2)-1) < 1e-8			
			assert np.abs(np.sum(morlet_values)) < 1e-8


			#~~~~~~~~~~~~
			#~~~PART 2~~~
			#~~~~~~~~~~~~
			im = imread(im_string)
			convolved_real_values = convolve2d(im, real_values, 'same')
			convolved_imaginary_values = convolve2d(im, imaginary_values, 'same')

			#add to global convolved arrays
			real_convolved_arrays.append(convolved_real_values)
			imaginary_convolved_arrays.append(convolved_imaginary_values)

			plt.figure(3)
			ax = plt.subplot(3,4,s_n*4+t_n+1)
			ax.set_title('σ:'+str(sigma)+' θ:'+theta_string_array[t_n])
			plt.imshow(convolved_real_values, cmap="gray")
			plt.tight_layout()
			
			plt.figure(4)
			ax = plt.subplot(3,4,s_n*4+t_n+1)
			ax.set_title('σ:'+str(sigma)+' θ:'+theta_string_array[t_n])
			plt.imshow(convolved_imaginary_values, cmap="gray")
			plt.tight_layout()
	all_images_real_convolved_arrays.append(real_convolved_arrays)
	all_images_imaginary_convolved_arrays.append(imaginary_convolved_arrays)

wavelet_edge_array = []
for im_index in range(0, len(images_to_convolve)):
	real_convolved_arrays = all_images_real_convolved_arrays[im_index]
	imaginary_convolved_arrays = all_images_imaginary_convolved_arrays[im_index]
	wavelet_edge_values = np.zeros((len(real_convolved_arrays[0]), len(real_convolved_arrays[0][0])))
	for x in range(0, len(real_convolved_arrays[0])):
		for y in range(0, len(real_convolved_arrays[0][0])):
			max_val = -10000
			for index in range(0, len(real_convolved_arrays)):
				val = abs(imaginary_convolved_arrays[index][x][y] - real_convolved_arrays[index][x][y])
				if(val > max_val):
					max_val = val
			wavelet_edge_values[x][y] = max_val

	wavelet_edge_array.append(wavelet_edge_values)
	plt.figure(5)
	ax = plt.subplot(1,len(images_to_convolve),im_index+1)
	ax.set_title(images_to_convolve[im_index])
	plt.imshow(wavelet_edge_values, cmap="gray")
	plt.tight_layout()




#~~~~~~~~~~~~
#~~~PART 2~~~
#~~~~~~~~~~~~
def error(x0, y0, disparity):
	epsilon = 0.001
	delta = 5
	sum_errors = 0
	for x_l in range(x0-delta, x0+delta+1):
		first_numerator = wavelet_edge_array[0][y0][x_l]+epsilon
		first_denominator = wavelet_edge_array[1][y0][x_l-disparity]+epsilon
		x = first_numerator/first_denominator
		sum_errors += x+(1/x)

	return sum_errors

'''
		first_numerator = wavelet_edge_array[0][y0][x_l]+epsilon
		second_denominator = first_numerator
		first_denominator = wavelet_edge_array[1][y0][x_l-disparity]+epsilon
		second_numerator = first_denominator

		sum_errors += ((first_numerator/first_denominator)+(second_numerator/second_denominator))
'''
height = len(wavelet_edge_array[0])
width = len(wavelet_edge_array[0][0])

error_minimizing_disp_values = np.zeros((height, width))
for y in range(0, len(error_minimizing_disp_values)):
	for x in range(0, len(error_minimizing_disp_values[y])):
		error_minimizing_disp_values[y][x] = -10

max_error = 0
for y_0 in range(30, height-30):
	for x_0 in range(30, width-30):
		min_error = 100000
		min_disp = 100000
		for disp in range(-4, 15):
			err = error(x_0, y_0, disp)
			if(err < min_error):
				min_error = err
				min_disp = disp
			if(err > max_error):
				max_error = err
		error_minimizing_disp_values[y_0][x_0] = min_disp


'''
error_minimizing_disp_values = np.zeros((height-60, width-60))
for y_0 in range(30, height-30):
	for x_0 in range(30, width-30):
		min_error = 100000
		min_disp = 100000
		for disp in range(-5, 15):
			err = error(x_0, y_0, disp)
			if(err < min_error):
				min_error = err
				min_disp = disp
		error_minimizing_disp_values[y_0-30][x_0-30] = min_disp
'''

plt.figure(6)
ax = plt.subplot(1,1,1)
ax.set_title("disparity values")
plt.imshow(error_minimizing_disp_values, cmap="gray")
plt.tight_layout()

#~~~~~~~~~~~~~~~~~~~~
#~~~PART 3~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~

height = len(wavelet_edge_array[0])
width = len(wavelet_edge_array[0][0])

disp_max = 15
disp_min = -5
occ = 0.18 * max_error

all_cost = []
all_prev = []

'''
for y in range(0, len(error_minimizing_disp_values)):
	for x in range(0, len(error_minimizing_disp_values[y])):
		best_cost[y][x] = 0
'''


x_l_start = 30
x_l_end = width-30-1

'''
#Forward Algorithm
for y_0 in range(30, height-30):
	best_cost = []
	best_prev = []
	for x_l in range(x_l_start, x_l_end+1):
		best_cost_arr = []
		best_prev_arr = []
		for x_r in range(x_l-disp_max, x_l-disp_min):
			first_val = occ+best_cost[x_l-1][x_r]
			second_val = occ+best_cost[x_l][x_r-1]
			third_val = error(x_l-1, y_0, x_l-x_r)+best_cost[x_l-1][x_r-1]
			vals = [first_val, second_val, third_val]
			args = [ [x_l-1, x_r], [x_l, x_r-1], [x_l-1, x_r-1] ]

			min_val = min(vals)
			min_index = vals.index(min_val)
			min_arg = args[min_index]

			best_cost_arr.append(min_val)
			best_prev_arr.append(min_arg)
		best_cost.append(best_cost_arr)
		best_prev.append(best_prev_arr)
	all_cost.append(best_cost)
	all_prev.append(best_prev)

disp_map = np.zeros((height, width))

for prev in all_prev:
	print(prev)


#Backtrack Algorithm
new_it = 0
for y_0 in range(30, height-30):
	best_cost = all_cost[new_it]
	best_prev = all_prev[new_it]
	
	xr_arr = best_cost[x_l_end-x_l_start]
	min_cost = 100000
	min_xr_val = 100000
	counter = 0
	for x_r in range(x_l_end-disp_max, x_l_end-disp_min):
		if(xr_arr[counter] < min_cost):
			min_cost = xr_arr[counter]
			min_xr_val = x_r
		counter+=1

	x_r_best = min_xr_val
	
	best_prev_point = best_prev[x_l_end-x_l_start][x_r_best-(x_l_end-disp_max)]

	while(best_prev_point[0] > x_l_start):
		best_prev_point = best_prev[ int(best_prev_point[0])-x_l][ int(best_prev_point[1]) ]
		disp = best_prev_point[0] - best_prev_point[1]
		disp_map[y_0][ int(best_prev_point[0]) ] = disp
		if(disp > 0):
			print("greater")

	new_it+=1
'''


#Forward Algorithm
for y_0 in range(30, height-30):
	best_cost = np.zeros((width, width))
	best_prev = np.zeros((width,  width, 2))
	for x_l in range(30, width-30):
		for x_r in range(x_l-disp_max, x_l-disp_min):
			first_val = occ+best_cost[x_l-1][x_r]
			second_val = occ+best_cost[x_l][x_r-1]
			third_val = error(x_l-1, y_0, x_l-x_r)+best_cost[x_l-1][x_r-1]
			vals = [first_val, second_val, third_val]
			args = [ [x_l-1, x_r], [x_l, x_r-1], [x_l-1, x_r-1] ]

			min_val = min(vals)
			min_index = vals.index(min_val)
			min_arg = args[min_index]

			best_cost[x_l][x_r] = min_val
			#print(str(min_arg[0])+" "+str(min_arg[1]))
			best_prev[x_l][x_r][0] = min_arg[0]
			best_prev[x_l][x_r][1] = min_arg[1]

	all_cost.append(best_cost)
	all_prev.append(best_prev)

disp_map = np.zeros((height, width))

#Backtrack Algorithm
new_it = 0
for y_0 in range(30, height-30):
	best_cost = all_cost[new_it]
	best_prev = all_prev[new_it]


	for x_l in range(x_l_start, x_l_end+1):
		output = ""
		for x_r in range(x_l-disp_max, x_l-disp_min):
			output = output+ " "+str( best_cost[x_l][x_r] )
		#print(output)

	xr_arr = best_cost[x_l_end]
	min_cost = 100000
	min_xr_val = 100000
	for x_r in range(x_l_end-disp_max, x_l_end-disp_min):
		if(xr_arr[x_r] < min_cost):
			min_cost = xr_arr[x_r]
			min_xr_val = x_r

		#print(str(best_prev_point[0])+" "+str(best_prev_point[1]) )
	x_r_best = min_xr_val
	#print(min_cost)	
	#print(str(x_l_end)+" "+str(x_r_best))
	
	output = ""
	for x_r in range(x_l_end-disp_max, x_l_end-disp_min):
		output += str(best_prev[x_l_end][x_r][0])+" "+str(best_prev[x_l_end][x_r][1])
	#print(output)

	best_prev_point = best_prev[x_l_end][x_r_best]
	#print(best_prev[ int(best_prev_point[0]) ] [int(best_prev_point[1])])
	#print(str(best_prev_point[0])+" "+str(best_prev_point[1]))
	while(best_prev_point[0] > x_l_start):
		#print(str(best_prev_point[0])+" "+str(best_prev_point[1]) )
		#print(str(int(best_prev_point[0]))+" "+str(int(best_prev_point[1])) )
		best_prev_point = best_prev[ int(best_prev_point[0]) ][ int(best_prev_point[1]) ]
		#print(str(best_prev_point[0])+" "+str(best_prev_point[1]) )
		disp = best_prev_point[0] - best_prev_point[1]
		#print(int(disp))
		#print(str(int(best_prev_point[0]))+" "+str(int(best_prev_point[1])) )
		disp_map[y_0][ int(best_prev_point[0]) ] = disp
		if(disp > 0):
			print("greater")
		if(disp < 0):
			print("lesser")

	new_it+=1


plt.figure(7)
ax = plt.subplot(1,1,1)
ax.set_title("disparity map")
plt.imshow(disp_map, cmap="gray")
plt.tight_layout()
plt.show()
