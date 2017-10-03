import numpy as np

class Lane(object):
    def __init__(self,x,y,fit_polynomial, fitx, fity):
        self.nonzerox = x
        self.nonzeroy = y
        self.fit_polynomial = fit_polynomial
        self.fitx = fitx
        self.fity = fity

    def closest_to_car(self):
        ind = np.argmax(self.fity)
        return self.fitx[ind]

    def calculate_curvature(self):
        y_eval = np.max(self.nonzeroy)

        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/570 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(self.nonzeroy*ym_per_pix, self.nonzerox*xm_per_pix, 2)
        # Calculate the new radii of curvature
        curve = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return int(curve)
