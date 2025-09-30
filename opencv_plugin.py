import cv2 as cv
import numpy as np

def hsl2bgr(h,s,l):
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
    
    bgr = np.stack([B, G, R], axis=-1)
    return bgr


def assemble_hsl_values(phases,intensity=False,log_scale=False,saturation=1):
    hue = phases%(2*np.pi)
    if isinstance(intensity,np.ndarray):
        value = np.sqrt(intensity)
        if log_scale:            
            value[value!=0] = np.log10(value[value!=0])
            vmin = value.min()
            value = (value-vmin)/(value.max()-vmin)    
        else:        
            value = (value/value.max())    
    else:        
        value = np.full_like(hue,intensity)
    lightness = value
    saturation = np.full_like(hue,saturation)
    hsl_array=np.stack((hue/(2*np.pi)*360.0, saturation,lightness),axis=-1)
    #hls_array=np.stack((hue,lightness, saturation),axis=-1)
    return hsl_array

def assemble_hls_values(phases,intensity=False,log_scale=False,saturation=1):
    hue = phases%(2*np.pi)
    if isinstance(intensity,np.ndarray):
        value = np.sqrt(intensity)
        if log_scale:            
            value[value!=0] = np.log10(value[value!=0])
            vmin = value.min()
            value = (value-vmin)/(value.max()-vmin)    
        else:        
            value = (value/value.max())    
    else:        
        value = np.full_like(hue,intensity)
    lightness = value
    saturation = np.full_like(hue,saturation)
    hls_array=(np.stack((hue/(2*np.pi),lightness, saturation),axis=-1)*255).astype(np.uint8)
    #hls_array=np.stack((hue,lightness, saturation),axis=-1)
    return hls_array

def get_phase_and_intensity(complex_array):
    abs_values = np.abs(complex_array).real
    non_zero_mask = (abs_values!=0)

    phases = np.full_like(abs_values,0)
    phases[non_zero_mask] = np.log(complex_array[non_zero_mask].astype(complex)/abs_values[non_zero_mask]).imag
    
    intensity = abs_values**2
    return phases,intensity

    
class CV_Plugin:
    colormaps={
        'autumn':cv.COLORMAP_AUTUMN,
        'bone':cv.COLORMAP_BONE,
        'jet':cv.COLORMAP_JET,
        'winter':cv.COLORMAP_WINTER,
        'rainbow':cv.COLORMAP_RAINBOW,
        'ocean': cv.COLORMAP_OCEAN,
        'summer': cv.COLORMAP_SUMMER,
        'spring': cv.COLORMAP_SPRING,
        'cool': cv.COLORMAP_COOL,
        'hsv': cv.COLORMAP_HSV,
        'pink': cv.COLORMAP_PINK,
        'hot': cv.COLORMAP_HOT,
        'parula': cv.COLORMAP_PARULA,
        'magma': cv.COLORMAP_MAGMA,
        'inferno': cv.COLORMAP_INFERNO,
        'plasma': cv.COLORMAP_PLASMA,
        'viridis': cv.COLORMAP_VIRIDIS,
        'cvidis': cv.COLORMAP_CIVIDIS,
        'twilight': cv.COLORMAP_TWILIGHT,
        'twilight_shfted': cv.COLORMAP_TWILIGHT_SHIFTED,
        'turbo': cv.COLORMAP_TURBO,
        'deep_green':cv.COLORMAP_DEEPGREEN
    }

    @classmethod
    def get_polar_image_data(cls,data,n_pixels=False):
        Nr,Nphi=data.shape
        pdata=data.copy()
        _min = pdata.min()
        if isinstance(n_pixels,bool):
            n_pixels=Nr*2
        radius = int(n_pixels//2)
        center = (int(n_pixels//2),int(n_pixels//2))
            
        pic = cv.warpPolar(pdata.T,center=center,maxRadius=radius,dsize=(n_pixels,n_pixels),flags=cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR)
        return pic
    

    @classmethod
    def get_polar_image(cls,data,n_pixels=False,scale='lin',colormap='viridis',vmin=False,vmax=False,print_colorscale=False,transparent_backgound=False):
        Nr,Nphi=data.shape
        pdata=data.copy()
        use_log_scale= (scale=='log')
        if use_log_scale:
            pdata[pdata<=0]=1e-15
            pdata=np.log10(pdata)         
        

        max_d = pdata.max()
        min_d = pdata.min()
        if isinstance(vmin,bool):
            vmin=min_d
        else:
            if use_log_scale:
                vmin=np.log10(np.abs(vmin))
            pdata[pdata<vmin]=vmin
            
        if isinstance(vmax,bool):
            vmax=max_d
        else:
            if use_log_scale:
                vmax=np.log10(np.abs(vmax))
            pdata[pdata>vmax]=vmax
        zero_in_range= (0>=vmin) and (0<=vmax)
        if zero_in_range:
            bg_value = 0
        else:
            bg_value = vmin

        enlarged_Nr = int(Nr*np.sqrt(2))
        temp_pic = np.zeros((enlarged_Nr,Nphi),dtype=float)
        bg_pic = np.zeros((enlarged_Nr,Nphi),dtype=float)
        temp_pic[:Nr]=pdata
        temp_pic[Nr:]=bg_value
        bg_pic[Nr:]=1
        
        if isinstance(n_pixels,bool):
            n_pixels=Nr*2
        radius = int(n_pixels/np.sqrt(2))
        center = (int(n_pixels//2),int(n_pixels//2))
            
        pic = cv.warpPolar(temp_pic.T,center=center,maxRadius=radius,dsize=(n_pixels,n_pixels),flags=cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR)
        bg_mask = cv.warpPolar(bg_pic.T,center=center,maxRadius=radius,dsize=(n_pixels,n_pixels),flags=cv.WARP_INVERSE_MAP + cv.WARP_POLAR_LINEAR)==1
        
        pic-=vmin
        pic*=255/(vmax-vmin)
        pic=pic.astype(np.uint8)
        pic= cv.applyColorMap(pic, cls.colormaps.get(colormap,cv.COLORMAP_VIRIDIS))
        if print_colorscale:
            h_black = int(pic.shape[0]*0.05)
            h_color_bar = int(pic.shape[0]*0.05)
            cscale=np.zeros((h_black+h_color_bar,pic.shape[0]),dtype=np.uint8)
            cscale[h_black:]=np.arange(pic.shape[0])*255/pic.shape[0]
            cscale = cv.applyColorMap(cscale, cls.colormaps.get(colormap,cv.COLORMAP_VIRIDIS))
            cscale[:h_black]=0

            if use_log_scale:
                vmin_txt = f"{np.exp(vmin):1.2e}"
                vmax_txt = f"{np.exp(vmax):1.2e}"
            else:
                vmin_txt = f"{vmin:1.2e}"
                vmax_txt = f"{vmax:1.2e}"
            WHITE = (255,255,255)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.03 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
            font_size = n_pixels/(25/scale)
            font_color = WHITE
            font_thickness = int(font_size*2)
            img_text = cv.putText(cscale, vmin_txt, (int(n_pixels*0),h_color_bar-int(h_black*0.2)), font, font_size, font_color, font_thickness, cv.LINE_AA)
            img_text = cv.putText(cscale, vmax_txt, (int(n_pixels*0.80),h_color_bar-int(h_black*0.2)), font, font_size, font_color, font_thickness, cv.LINE_AA)
            pic=np.concatenate((pic,cscale),axis=0)
            

        if transparent_backgound:
            # First create the image with alpha channel
            pic_rgba = cv.cvtColor(pic, cv.COLOR_RGB2RGBA)

            # Then assign the mask to the last channel of the image
            pic_rgba[:n_pixels, :n_pixels, 3] = ((~bg_mask).astype(np.uint8)*255)
            pic=pic_rgba.copy()

        return pic

    
    @classmethod
    def save(cls,path,image,colors_type = 'bw'):
        
        #path=path.rsplit[:-3]+'.png'
        compression_params = [cv.IMWRITE_TIFF_COMPRESSION, 1] 
        cv.imwrite(path, image,compression_params)
        cv.waitKey(0)

    @classmethod
    def save_hls(cls,path,hue,intensity=False,log_scale=False,saturation=1):
        hls_array = assemble_hls_values(hue,intensity=intensity,log_scale=log_scale,saturation=saturation)
        bgr_array = cv.cvtColor(hls_array,cv.COLOR_HLS2BGR_FULL)
        #bgr_array = (hls_to_bgr(hls_array)*255).astype(np.uint8)
        cls.save(path,bgr_array)

    @classmethod
    def save_hsl(cls,path,hue,intensity=False,log_scale=False,saturation=1):
        print(hue.dtype)
        hsl_array = assemble_hsl_values(hue,intensity=intensity,log_scale=log_scale,saturation=saturation)
        print(hsl_array.dtype)
        bgr_array = hsl2bgr(hsl_array[...,0],hsl_array[...,1],hsl_array[...,2])
        print(hsl_array.dtype)
        bgr_array = bgr_array.astype(np.float32)
        print(bgr_array.shape,bgr_array.dtype)
        #bgr_array = (hls_to_bgr(hls_array)*255).astype(np.uint8)
        cls.save(path,bgr_array.astype(np.float32))
        
    @classmethod
    def save_complex(cls,path,image,log_scale=False,saturation=1):
        phases,intensity= get_phase_and_intensity(image)
        cls.save_hsl(path,phases,intensity = intensity,log_scale = log_scale,saturation=saturation)
        print(path)
    @classmethod
    def load(cls,path,as_grayscale=False):
        print(f'Loading Image from: {path}')
        image = cv.imread(path)
        cv.waitKey(0)
        if as_grayscale:
            image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        return image.astype(float)
    
    @classmethod
    def save_video_complex(cls,path,images,log_scale=False,saturation=1):
        # choose codec according to format needed
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        image_size=images.shape[1:]
        video = cv.VideoWriter(path, fourcc, 10, image_size)
        for im in images:
            hls_img = assemble_hls_values(np.zeros_like(im),intensity=im,log_scale=log_scale,saturation=saturation)
            bgr_img = cv.cvtColor(hls_img,cv.COLOR_HLS2BGR)
            video.write(bgr_img)
        cv.destroyAllWindows()
        video.release()

    @classmethod
    def create_image(cls,data,log_scale=False,colormap='viridis',vmin=False,vmax=False):
        data=data.copy()
        use_log_scale= log_scale
        if use_log_scale:
            data[data<=0]=1e-15
            data=np.log10(data)
            
        max_d = data.max()
        min_d = data.min()
        if isinstance(vmin,bool):
            vmin=min_d
        else:
            if use_log_scale:
                vmin=np.log10(np.abs(vmin))
            data[data<vmin]=vmin
            
        if isinstance(vmax,bool):
            vmax=max_d
        else:
            if use_log_scale:
                vmax=np.log10(np.abs(vmax))
            data[data>vmax]=vmax
        zero_in_range= (0>=vmin) and (0<=vmax)
        if zero_in_range:
            bg_value = 0
        else:
            bg_value = vmin
        data-=vmin
        if (vmax-vmin)!=0:
            data*=255/(vmax-vmin)
        else:
            data[:]=0
        data=data.astype(np.uint8)
        data= cv.applyColorMap(data, cls.colormaps.get(colormap,cv.COLORMAP_VIRIDIS))

        if True:
            h_black = int(data.shape[0]*0.05)
            h_color_bar = int(data.shape[0]*0.05)
            cscale=np.zeros((h_black+h_color_bar,data.shape[0]),dtype=np.uint8)
            cscale[h_black:]=np.arange(data.shape[0])*255/data.shape[0]
            cscale = cv.applyColorMap(cscale, cls.colormaps.get(colormap,cv.COLORMAP_VIRIDIS))
            cscale[:h_black]=0

            if use_log_scale:
                vmin_txt = f"{np.exp(vmin):1.2e}"
                vmax_txt = f"{np.exp(vmax):1.2e}"
            else:
                vmin_txt = f"{vmin:1.2e}"
                vmax_txt = f"{vmax:1.2e}"
            WHITE = (255,255,255)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.03 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
            n_pixels=data.shape[0]
            font_size = n_pixels/(25/scale)
            font_color = WHITE
            font_thickness = int(font_size*2)
            img_text = cv.putText(cscale, vmin_txt, (int(n_pixels*0),h_color_bar-int(h_black*0.2)), font, font_size, font_color, font_thickness, cv.LINE_AA)
            img_text = cv.putText(cscale, vmax_txt, (int(n_pixels*0.80),h_color_bar-int(h_black*0.2)), font, font_size, font_color, font_thickness, cv.LINE_AA)
            data=np.concatenate((data,cscale),axis=0)
        return data
        
    @classmethod
    def save_video(cls,path,images,log_scale=False,colormap='inferno'):
        # choose codec according to format needed
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        image_size=cls.create_image(images[0],log_scale=log_scale,colormap=colormap).shape[:2]
        #print(image_size)
        video = cv.VideoWriter(path, fourcc, 60, image_size[::-1])
        for im in images:
            bgr_img = cls.create_image(im,log_scale=log_scale,colormap=colormap)
            #print(bgr_img.shape)
            video.write(bgr_img)
        cv.destroyAllWindows()
        video.release()
