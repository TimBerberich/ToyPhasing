import os
import time
from os import path as op
from pathlib import Path
import numpy as np
from opencv_plugin import CV_Plugin,get_phase_and_intensity
from images import load,save_complex,save_hsl


#  ###################################################
#  ## Traditional iterative phase retrieval scheme ###
#  ###################################################
#
#                 #####################                  
#        ---->----# Fourier Transform #---->----
#        |        #####################        |
#        |                                     |
#        |                                     |
#  ##############                        ##############
#  #   Support  #                        # Intensity  #
#  # Projection #                        # Projection #
#  ##############                        ##############
#        |                                     | 
#        |                                     |
#        |         ###################         |
#        -----<----# Inverse Fourier #---<------
#                  #   Transform     #
#                  ###################
#
# Starting form a random density guess 'd', the correct density is approximated
# using the following process:
# 1. Calculate the fourier Transform F(d) of d.
# 2. Enforce the known intensity values (I) by setting F(d2) = F(d)/|F(d)| * sqrt(I)
#    This is the Intensity projection
# 3. Inverse fourier transform F(d2) to obtain the new density guess d2
# 4. Project all values of d2 outside of the density support to 0 and call the new density d3
# 5.
#    a) In case of Error reduction (ER) iterations continue in step 1. with d3
#    b) In case of Error reduction (HIO) iterations continue in step 1. with d4 = d - b*(d3-d2)

# Every couple of iterations it helps to update the support region by a method called shrink-wrap.
# Here the current density guess is blurred and the new support area is defined by the set of
# blurred density values that are above some threshold relative to the maximum of the
# blurred density.

# This phasing algorithm tries to optimize a random input function such that it simultaneousely satisfies
# a support constraint and the constraint that its fourier transform has given absolute values I.

# This is a high dimensional optimization problem and there do not exist convergence guaranties.
# In fact the problem can be understood as special case of quadratic programming 
# (https://en.wikipedia.org/wiki/Quadratic_programming) which is NP hard.
# It can happen that for certain inputs the algorithm gets stuck in local minima. The HIO part of
# the algorithm tries to escape local minima by providing negative feedback where ever the support
# condition is not satisfied. (In step 5. b) b*(d3-d2) is nonzero only were the support constraint
# not sattisfied. The constant b regulate the strength of the negative feedback and can take values
# between 0 an 1.)
#
# Some useful references on phasing algorithms are:
# Fienup, J. R. (1978), ‘Reconstruction of an object from the modulus of its fourier transform’, Opt. Lett. 3(1), 27–29.
# Marchesini et al. (2003), ‘X-ray image reconstruction from a diffraction pattern alone’, Phys. Rev. B 68(14).

def hio_projection(density,support_mask,initial_density,hio_parameter):
    density = np.where(support_mask,density,initial_density-hio_parameter*density) 
    return density

def er_projection(density,support_mask):
    density[~support_mask]=0
    return density

def generate_ER_loop(intensity):
    empty_density = np.zeros_like(intensity)
    def er_loop(density,mask):
        ft_density = np.fft.fft2(density)
        non_zero_mask = (ft_density!=0)
        projected_ft_density = empty_density.copy()
        projected_ft_density[non_zero_mask] = ft_density[non_zero_mask]/np.abs(ft_density[non_zero_mask])*np.sqrt(intensity[non_zero_mask])
        #projected_ft_density[non_zero_mask] = ft_density[non_zero_mask]
        new_density = np.fft.ifft2(projected_ft_density)
        projected_density = er_projection(new_density,mask)
        projected_density[projected_density<0]=0
        return projected_density
    return er_loop

def generate_HIO_loop(intensity,hio_parameter):
    empty_density = np.zeros_like(intensity)
    def HIO_loop(density,mask):
        ft_density = np.fft.fft2(density)
        non_zero_mask = (ft_density!=0)
        projected_ft_density = empty_density.copy()
        projected_ft_density[non_zero_mask] = ft_density[non_zero_mask]/np.abs(ft_density[non_zero_mask])*np.sqrt(intensity[non_zero_mask])
        new_density = np.fft.ifft2(projected_ft_density)
        projected_density = hio_projection(new_density,mask,density,hio_parameter)
        projected_density[projected_density<0]=0
        return projected_density
    return HIO_loop

def generate_shrink_wrap(threshold,sigma,data_shape):
    '''
    This method first creates a blurred version of the input density by convolving the density with a gaussian distriution of given standard deviation sigma.
    This is achived by using the convolution theorem of fourier transforms and the fact that the fourier transform of a gaussian distribution has an exact expression (ft_gaussian).
    Afterwards a new support mast is created by thresholding the blurred density.
    '''
    q_x = np.fft.fftfreq(data_shape[0])
    q_y = np.fft.fftfreq(data_shape[1])
    abs_squared_q = q_x[:,None]**2+q_y[None,:]**2
    a = 1/(2*sigma**2)
    ft_gaussian = np.sqrt(np.pi/a)*np.exp(-np.pi**2*abs_squared_q/a)
    def SW(density):
        ft_density = np.fft.fft2(density)
        ft_density *= ft_gaussian
        blurry_density = np.fft.ifft2(ft_density)
        max_density_value = blurry_density.max()
        mask = (blurry_density >= max_density_value*threshold)
        return mask
    return SW

def density_error(density,new_density):
    '''Estimates the reconstruciton error by simply summing the density outside of the support region
       relative to the total summed density.'''
    a=density.real
    b=new_density.real
    error = np.sqrt(np.sum((a-b)**2))/np.sqrt(np.sum(a**2))
    return error

def get_history_parameters(intensity,loop_parameters,max_RAM = 1):
    N_steps = loop_parameters[0]*(loop_parameters[1]+loop_parameters[2])+loop_parameters[3]
    image_size = np.prod(intensity.shape)*np.dtype(float).itemsize
    max_history_size = image_size*N_steps
    max_RAM_Bytes = max_RAM*1024**3
    if max_RAM_Bytes >= max_history_size:
        history_stride=1
        history_length=N_steps
    else:
        history_stride = max_history_size//max_RAM_Bytes +1
        history_length=N_steps//history_stride+1
    #print(f'image shape = {intensity.shape} data size MB = {image_size/(1024**2)}')
    #print(f'stride {history_stride} size = {history_length}, max ram bytes // history size {max_history_size//max_RAM_Bytes}')
    return history_stride,history_length

def assemble_phasing(initial_mask,intensity,hio_parameter,sw_threshold,sw_sigma,loop_parameters,max_RAM=1):
    loop_iterations = loop_parameters[0]
    ER_iterations = loop_parameters[1]
    HIO_iterations = loop_parameters[2]
    #hio_parameter = 0.5
    ER_refinement_iterations = loop_parameters[3]
    
    ER_loop = generate_ER_loop(intensity)
    HIO_loop = generate_HIO_loop(intensity,hio_parameter)
    SW = generate_shrink_wrap(sw_threshold,sw_sigma,intensity.shape)
    mask = initial_mask

    history_stride,history_length = get_history_parameters(intensity,loop_parameters,max_RAM=max_RAM)
    history = np.zeros((history_length,)+(intensity.shape))
    #print(f'history shape = {history.shape} stride = {history_stride}')
    def phasing_loop(density_guess):
        density = density_guess
        mask = initial_mask
        step_counter=0
        errors=[]
        for i in range(loop_iterations):
            for hio_i in range(HIO_iterations):
                new_density = HIO_loop(density,mask)
                error = density_error(density,new_density)
                density = new_density
                
                if step_counter%history_stride==0:
                    history[step_counter//history_stride]=density.real
                errors.append(error)
                step_counter+=1
            if HIO_iterations>0:
                print('Loop {}: HIO error = {}'.format(i,error))
            mask = SW(density.real)#*initial_mask
            for er_i in range(ER_iterations):
                new_density = ER_loop(density,mask)
                error = density_error(density,new_density)
                density = new_density
                
                if step_counter%history_stride==0:
                    history[step_counter//history_stride]=density.real
                errors.append(error)
                step_counter+=1
            if ER_iterations>0:
                print('Loop {}: ER error = {}'.format(i,error))
            
        for er_i in range(ER_refinement_iterations):
            new_density = ER_loop(density,mask)
            error = density_error(density,new_density)
            density = new_density
            
            if step_counter%history_stride==0:
                history[step_counter//history_stride]=density.real
            errors.append(error)
            step_counter+=1
        print('Loop {}: Final error = {}'.format(i,error))
        return density,mask,history,errors

    return phasing_loop


def define_file_paths(image_path,mask_path):   
    name = image_path.name
    filetype = name.split('.')[-1]
    name=name[:-len(filetype)-1]
    time_struct=time.gmtime()
    time_str=str(time_struct[2])+'_'+str(time_struct[1])+'_'+str(time_struct[0])+'-'+str(time_struct[3])+'_'+str(time_struct[4])+'_'+str(time_struct[5])
    output_folder = (base_path / f'./results/{name}/{time_str}/').resolve()
    fft_images = (output_folder / './fft_images/').resolve()
    if not output_folder.exists():
        os.makedirs(output_folder.as_posix())
        os.makedirs(fft_images.as_posix())
    
    output_path_fft_image = (fft_images / f'fft.tiff').resolve().as_posix()
    output_path_intensity_image = (fft_images / f'fft_intensity.tiff').resolve().as_posix()
    output_path_phase_image = (fft_images / f'fft_phase.tiff').resolve().as_posix()
    output_path_intensity_inverse_image = (fft_images / f'sqrt_of_intensity_inverse.tiff').resolve().as_posix()
    output_path_autocorrelation_image = (fft_images /  f'autocorrelation.tiff').resolve().as_posix()
    output_path_phase_inverse_image = (fft_images / f'phase_inverse.tiff').resolve().as_posix()
    output_path_inverse_image = (fft_images / f'full_inverse.tiff').resolve().as_posix()

    initial_density_path = ( output_folder / f'initial_density.tiff').resolve().as_posix()
    initial_mask_path = ( output_folder / f'initial_mask.tiff').resolve().as_posix()
    reconstruction_path = ( output_folder/ f'reconstructed_density.tiff').resolve().as_posix()
    reconstructed_intensity_path = ( output_folder/ f'reconstructed_intensity.tiff').resolve().as_posix()
    reconstruction_video_path = (output_folder /  f'reconstruction_video_density.avi').resolve().as_posix()
    reconstruction_video_path_color = ( output_folder / f'reconstruction_video_density_colored.avi').resolve().as_posix()
    reconstructed_mask_path = (output_folder / 'reconstructed_mask.tiff').resolve().as_posix()
    
    return locals()
def save_images(phases,intensity,inverse_array,intensity_inverse_array,phases_inverse_array,autocorrelation):
    fft_array=np.sqrt(intensity)*np.exp(1.j*phases)

    ## Fourier transformed Image
    # Save Fouriertransform image 
    save_complex(output_path_fft_image,fft_array,log_scale=True)
    save_complex(output_path_intensity_image,intensity,saturation=0,log_scale=True)
    save_hsl(output_path_phase_image,phases,intensity=0.5,saturation=1)
    
    
    ## Inverses of fourier transformed Image
    # Save inverse constructed from square root of intensity only 
    save_complex(output_path_intensity_inverse_image,intensity_inverse_array,saturation=0,log_scale=False)
    # Save inverse constructed from of intensity only (This is the autocorrelation)
    save_complex(output_path_autocorrelation_image,autocorrelation,saturation=0,log_scale=False)
    # Save inverse constructed from phases only 
    save_complex(output_path_phase_inverse_image,phases_inverse_array,saturation=0,log_scale=True)
    # Save inverse of complete fourier transform
    save_complex(output_path_inverse_image,inverse_array,saturation=0,log_scale=False)
    
def density_to_fft_intensity(density):
    ft_density = np.fft.fft2(density)
    intensity = ft_density*ft_density.conj()
    return intensity

def fft_example(input_is_intensity=False):
    if input_is_intensity:
        intensity = (load(image_path,as_grayscale=True))
        intensity/=intensity.max()
        print(intensity.dtype,intensity.max())
        nr = len(intensity)//2
        intensity = np.roll(np.roll(intensity,nr,axis =0),nr,axis = 1)
        autocorrelation = np.abs(np.fft.ifft2(intensity.real))

        intensity = np.roll(np.roll(intensity,nr,axis =0),nr,axis = 1)
        autocorrelation = np.roll(np.roll(autocorrelation,nr,axis =0),nr,axis = 1)
        save_complex(output_path_intensity_image,intensity,saturation=0,log_scale=False)
        save_complex(output_path_autocorrelation_image,autocorrelation,saturation=0,log_scale=False)
    else:
        # load image  | Lade Bild datei
        bw_array = load(image_path,as_grayscale=True)
        # Fouriertransform Image 
        fft_array = np.fft.fft2(bw_array)
        # Phases and Intensity of fourier transformed image 
        phases, intensity = get_phase_and_intensity(fft_array)
        # inverse transform of intensities only
        intensity_inverse_array = np.abs(np.fft.ifft2(np.sqrt(intensity.real)))
        autocorrelation = np.abs(np.fft.ifft2(intensity.real))
        # inverse transform of phases only 
        phases_inverse_array = np.fft.ifft2(np.exp(1.j*phases))
        # inverse transform of complete data 
        inverse_array = np.fft.ifft2(fft_array)
        nr = len(bw_array)//2
        phases = np.roll(np.roll(phases,nr,axis =0),nr,axis = 1)
        intensity = np.roll(np.roll(intensity,nr,axis =0),nr,axis = 1)
        autocorrelation = np.roll(np.roll(autocorrelation,nr,axis =0),nr,axis = 1)
        intensity_inverse_array = np.roll(np.roll(intensity_inverse_array,nr,axis =0),nr,axis = 1)
        save_images(phases,intensity,inverse_array,intensity_inverse_array,phases_inverse_array,autocorrelation)    
    




if __name__ == '__main__':
    '''
    specify an input image in the code below and start the script via
    pyton simple_phasing.py in your console
    '''
    print('----- start processing ------\n')
    base_path = Path(__file__).parent

    #image_path =  (base_path / './disk_intensity_small.png').resolve()
    image_path =  (base_path / './disk_intensity_sim.tiff').resolve()
    #image_path =  (base_path / './disk_small.png').resolve()
    mask_path = (base_path / './disk_small_mask.png').resolve()
    locals().update(define_file_paths(image_path,mask_path))
    image_path =  image_path.as_posix()
    mask_path = mask_path.as_posix()
    #### Start Computations ####
    calc_fft_images = True
    input_is_intensity = True
    do_phasing = False
    
    if calc_fft_images:
        print('Calculating Fourier Transform Images')
        fft_example(input_is_intensity)

    if do_phasing:
        print('Start Phase retrieval:')
        #### Input parameters for the phase retrieval process ####

        # hio_parameter
        # Regulates negative feedback strength in HIO iterations.
        # Sensible values are between 0 and 1 commonly 0.5 is used.
        hio_parameter = 1 #1.0

        # Parameters for the shrink wrap routine
        
        # sigma
        # Defines the standard deviation of the gaussian burring filter.
        # A sigma value of 1 defines the burred desnity as convolution of the input density with a gaussian distribution that has a standard deviation of 1 pixel.
        sigma = 2
        
        # threshold
        # Regulates the area which is considered as new function support.
        # Values are between 0 and 1.
        # A value of e.g. 0.15 indicates that the new support area is defined by all pixels of the blurred density that have values higher or equal to 15% of the maximal blurred density value. 
        threshold = 0.1

        # Phasing loop parameters
        
        # Number of overall phasing loop iterations.
        loop_iterations = 4
        # Number of Error Reduction (ER) steps in each loop iteration.
        ER_iterations = 20
        # Number if Hybrid Input-Output steps in each loop iteration.
        HIO_iterations = 145
        # Number of final Error Reduction (ER) steps after all loop iterations are finished.
        ER_refinement_iterations = 200
        
        loop_parameters = [loop_iterations,ER_iterations,HIO_iterations,ER_refinement_iterations]

        max_RAM = 10 # In Gigabyte
        

        if input_is_intensity:
            intensity = load(image_path,as_grayscale=True)
            nr = len(intensity)//2
            intensity = np.roll(np.roll(intensity,nr,axis =0),nr,axis = 1)
            
            x_len,y_len = intensity.shape
            initial_mask = load(mask_path,as_grayscale=True)
            initial_mask = (initial_mask!=0)
            save_complex(initial_mask_path,initial_mask.astype(float))        
        else:
            density = CV_load(image_path,as_grayscale=True)
            x_len,y_len = density.shape
            initial_mask = load(mask_path,as_grayscale=True)
            initial_mask = (initial_mask!=0)
            save_complex(initial_mask_path,initial_mask.astype(float))        
            intensity=density_to_fft_intensity(density)    
                    
        phasing = assemble_phasing(initial_mask,intensity.astype(complex),hio_parameter,threshold,sigma,loop_parameters,max_RAM=max_RAM)
    
        initial_density = (1+0.1*np.random.rand(*intensity.shape))
        initial_density[~initial_mask]=0
        save_complex(initial_density_path,initial_density)
    
        reconstruction,final_mask,history,errors = phasing(initial_density)

        print('Saving phasing results.')
        reconstruction = reconstruction.real
        non_zero= reconstruction.real<0
        reconstruction[non_zero] = np.log(reconstruction[non_zero])

        nr = len(reconstruction)//2
        intensity = np.roll(np.roll(np.abs(np.fft.fft2(reconstruction))**2,nr,axis =0),nr,axis = 1)
        
        save_complex(reconstruction_path,reconstruction,saturation=0)
        save_complex(reconstructed_intensity_path,intensity,saturation=0,log_scale=True)
        save_complex(reconstructed_mask_path,final_mask.astype(float),saturation=0)
        #CV_Plugin.save_video_complex(reconstruction_video_path,history,saturation=0,log_scale=False)
        CV_Plugin.save_video(reconstruction_video_path_color,history,log_scale=False,colormap='jet')
        print('Done!')

    print('----- Stop processing ------')
