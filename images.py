import numpy as np
import skimage

def hsl2rgb(h,s,l):
    # intermediate values (Wikipedia)
    Hp = h / 60.0                             # H'
    C = (1.0 - np.abs(2.0 * l - 1.0)) * s     # chroma
    X = C * (1.0 - np.abs(np.mod(Hp, 2.0) - 1.0))
    
    # piecewise assignment for (R1, G1, B1)
    conds = [
        (0.0 <= Hp) & (Hp < 1.0),
        (1.0 <= Hp) & (Hp < 2.0),
        (2.0 <= Hp) & (Hp < 3.0),
        (3.0 <= Hp) & (Hp < 4.0),
        (4.0 <= Hp) & (Hp < 5.0),
        (5.0 <= Hp) & (Hp < 6.0),
    ]
    
    R1_choices = [C, X, 0.0, 0.0, X, C]
    G1_choices = [X, C, C, X, 0.0, 0.0]
    B1_choices = [0.0, 0.0, X, C, C, X]
    
    R1 = np.select(conds, R1_choices, default=0.0)
    G1 = np.select(conds, G1_choices, default=0.0)
    B1 = np.select(conds, B1_choices, default=0.0)
    
    # match lightness
    m = l - C / 2.0
    R = R1 + m
    G = G1 + m
    B = B1 + m
    
    rgb = np.stack([R, G, B], axis=-1)
    return rgb
def assemble_hsl_values(phases,intensity=False,log_scale=False,saturation=1):
    hue = phases%(2*np.pi)
    if isinstance(intensity,np.ndarray):
        value = np.sqrt(intensity)
        if log_scale:
            zero_mask = value==0
            value[value!=0] = np.log10(value[value!=0])
            vmin = value.min()
            vmax = value.max()
            value = (value-vmin)/(vmax-vmin)
            #value/=vmax
            value[zero_mask]=1            
        else:
            #print(value.min())
            value = (value/value.max())    
    else:        
        value = np.full_like(hue,intensity)
    lightness = value
    saturation = np.full_like(hue,saturation)
    hsl_array=np.stack((hue/(2*np.pi)*360.0, saturation,lightness),axis=-1)
    #hls_array=np.stack((hue,lightness, saturation),axis=-1)
    return hsl_array
def get_phase_and_intensity(complex_array):
    abs_values = np.abs(complex_array).real
    non_zero_mask = (abs_values!=0)

    phases = np.full_like(abs_values,0)
    phases[non_zero_mask] = np.log(complex_array[non_zero_mask].astype(complex)/abs_values[non_zero_mask]).imag
    
    intensity = abs_values**2
    return phases,intensity

def save(path,image):
    skimage.io.imsave(path,image)
def load(path,as_grayscale=False,bit_depth=None):
    image = skimage.io.imread(path)
    if as_grayscale:
        scale = 1.0
        if image.dtype == np.uint8:
            scale = 255
        elif image.dtype == np.uint16:
            scale = 65535
        elif image.dtype == np.uint32:
            scale = 4294967295

        image = image.astype(float)
        if image.ndim >2:
            image = np.sum(image[...,:3],axis=-1)/(3*scale)
        else:
            image/=scale
    image[np.isnan(image)]==1e-16
    if isinstance(bit_depth,int):
        scale=2**bit_depth-1
        image*=scale
        image[:] = (image//1)/scale
    return image
def save_hsl(path,hue,intensity=False,log_scale=False,saturation=1):
    hsl_array = assemble_hsl_values(hue,intensity=intensity,log_scale=log_scale,saturation=saturation)
    rgb_array = hsl2rgb(hsl_array[...,0],hsl_array[...,1],hsl_array[...,2])
    rgb_array = rgb_array.astype(np.float32)
    save(path,rgb_array)

def save_complex(path,image,log_scale=False,saturation=1):
    phases,intensity= get_phase_and_intensity(image)
    save_hsl(path,phases,intensity = intensity,log_scale = log_scale,saturation=saturation)
