"""
Document containing all functions in the pipeline for which stellar
and planetary data from the NASA Exoplanet Archive will be passed through.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import lightkurve as lk
from PIL import Image
from os.path import join
from glob import glob
from astropy.io import fits
import astrobase as ab
from astrobase.plotbase import skyview_stamp
from astropy.wcs import WCS
import os
from copy import deepcopy
from datetime import datetime
#from astrobase.periodbase.zgls import pgen_lsp

AUTHOR_DURATION = {"SPOC": 120, "QLP": 'long'}
FLUX_REFERENCE = {"SPOC": "pdcsap_flux", "QLP": "kspsap_flux"}
FLUX_LABEL = {"SPOC": "PDCSAP Flux", "QLP": "KSPSAP Flux"}


def HJ_scatter(min_p_orb = 0, max_p_orb = 10, 
                       min_rad = 8, max_rad = 30, 
                       min_mass = 95, max_mass = 6356):
    """
    Function returning a plot of the distribution of orbital periods versus
    planetary radii for a specified subset of the NASA Exoplanet Archive's 
    Planetary Systems Composite Data. 
    
    Parameters:
        - min_p_orb (float): minimum orbital period in days
        - max_p_orb (float): maximum orbital period in days
        - min_rad (float): minimum radius in Earth radii
        - max_rad (float): maximum radius in Earth radii
        - min_mass (float): minimum mass in Earth masses
        - max_mass (float): maximum mass in Earth masses

    Returns:
        - p_orb_rad_scatterplot (figure object): A plot of the HJ candidates for 
        orbital period versus radius given the specified parameters.
        - p_orb_mass_scatterplot (figure object): A plot of the HJ candidates for 
        orbital period versus mass given the specified parameters.
    """
    df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#") # accessing csvfile as dataframe
    sel = ((df["pl_orbper"]>=min_p_orb) & (df["pl_orbper"]<=max_p_orb) 
           & (df["pl_rade"]>=min_rad) & (df["pl_rade"]<=max_rad) 
           & (df["pl_bmasse"]>=min_mass) & (df["pl_bmasse"]<=max_mass))
    df_sample = df[sel]
    fp_orb = df_sample["pl_orbper"]
    fr_e = df_sample["pl_rade"]
    fm_e = df_sample["pl_bmasse"]

    # plot radius
    fig, ax = plt.subplots(figsize=(10,6))
    plt.grid(visible=True, which='both', axis='both')
    ax.scatter(fp_orb, fr_e, marker='.', color='k', s=7, alpha=1) # s=size | opacity alpha=0<x<1
    ax.set_xlabel('Orbital Period [Days]', fontsize=20)
    ax.set_ylabel('Radius [Earth Radii]', fontsize=20)
    ax.set_title("All planet candidates with " + str(min_mass) + " < m < " + str(max_mass) + r" $M_{Earth}$, " + 
                 str(min_rad) + " < r < " + str(max_rad) + r" $r_{Earth}$, and " + 
                 str(min_p_orb) + r" < $p_{orb}$ < " + str(max_p_orb) + " days", fontsize = 14)
    p_orb_rad_scatterplot = plt.show()

     # plot mass
    fig, ax = plt.subplots(figsize=(10,6))
    plt.grid(visible=True, which='both', axis='both')
    ax.scatter(fp_orb, fm_e, marker='.', color='k', s=7, alpha=1) # s=size | opacity alpha=0<x<1
    ax.set_xlabel('Orbital Period [Days]', fontsize=20)
    ax.set_ylabel('Mass [Earth Masses]', fontsize=20)
    ax.set_title("All planet candidates with " + str(min_mass) + " < m < " + str(max_mass) + r" $M_{Earth}$, " + 
                 str(min_rad) + " < r < " + str(max_rad) + r" $r_{Earth}$, and " + 
                 str(min_p_orb) + r" < $p_{orb}$ < " + str(max_p_orb) + " days", fontsize = 14)
    p_orb_mass_scatterplot = plt.show()

    return p_orb_rad_scatterplot, p_orb_mass_scatterplot
    

def HJ_scatter_all_color(min_p_orb = 0, max_p_orb = 10, 
                       min_rad = 8, max_rad = 30, 
                       min_mass = 95, max_mass = 6356, brightness = 13.5):
    """
    Function returning a plot of the distribution of orbital periods versus
    planetary radii and orbital periods versus planetary masses for a 
    specified subset of the NASA Exoplanet Archive's Planetary Systems Composite Data.
    
    Parameters:
        - min_p_orb (float): minimum orbital period in days
        - max_p_orb (float): maximum orbital period in days
        - min_rad (float): minimum radius in Earth radii
        - max_rad (float): maximum radius in Earth radii
        - min_mass (float): minimum mass in Earth masses
        - max_mass (float): maximum mass in Earth masses

    Returns:
        - p_orb_scatterplot (figure object): A plot of the HJ candidates for 
        orbital period versus radius and orbital period vs mass given the specified parameters.
    """
    df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#") # accessing csvfile as dataframe
    sel = ((df["pl_orbper"]>=min_p_orb) & (df["pl_orbper"]<=max_p_orb) 
        & (df["pl_rade"]>=min_rad) & (df["pl_rade"]<=max_rad) 
        & (df["pl_bmasse"]>=min_mass) & (df["pl_bmasse"]<=max_mass))
    df_sample = df[sel]
    fp_orb = df_sample["pl_orbper"]
    fr_e = df_sample["pl_rade"]
    fm_e = df_sample["pl_bmasse"]

    magsel = sel & (df["sy_tmag"]< brightness)
    df_mag = df[magsel]

    magfp_orb = df_mag["pl_orbper"]
    magfr_e = df_mag["pl_rade"]
    magfm_e = df_mag["pl_bmasse"]

    # plot mass
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    ax1.grid(visible=True, which='both', axis='both')
    ax1.scatter(fp_orb, fm_e, marker='.', color='#FD6A02', s=7, alpha=1, label='TESS mag ≥ ' + str(brightness)) # s=size | opacity alpha=0<x<1
    ax1.scatter(magfp_orb, magfm_e, marker='.', color='#1035AC', s=7, label='TESS mag < ' + str(brightness))
    ax1.set_yscale('log')
    #ax1.set_xlabel('Orbital Period [Days]', fontsize=20)
    #ax1.legend(['TESS mag ≥ 13.5', 'TESS mag < 13.5'])
    ax1.set_ylabel('Mass [Earth Masses]', fontsize=12)
    ax1.legend()
    ax1.set_title("All planet candidates with " + str(min_mass) + " < m < " + str(max_mass) + r" $M_{Earth}$, " + "\n" +
                str(min_rad) + " < r < " + str(max_rad) + r" $r_{Earth}$, and " + 
                str(min_p_orb) + r" < $p_{orb}$ < " + str(max_p_orb) + " days", fontsize = 14)
    ax2.scatter(fp_orb, fr_e, marker='.', color='#FD6A02', s=7, alpha=1) # s=size | opacity alpha=0<x<1
    ax2.scatter(magfp_orb, magfr_e, marker='.', color='#1035AC', s=7)
    ax2.grid(visible=True, which='both', axis='both')
    ax2.set_xlabel('Orbital Period [Days]', fontsize=12)
    ax2.set_ylabel('Radius [Earth Radii]', fontsize=12)
    #ax2.set_title("All planet candidates with " + str(min_mass) + " < m < " + str(max_mass) + r" $M_{Earth}$, " + 
    #                str(min_rad) + " < r < " + str(max_rad) + r" $r_{Earth}$, and " + 
    #                str(min_p_orb) + r" < $p_{orb}$ < " + str(max_p_orb) + " days", fontsize = 10)
    plt.savefig('HJ_scatter_color' + str(min_p_orb) + '.' + str(max_p_orb) + '.' + str(min_mass) + '.' +
                str(max_mass) + '.' + str(min_rad) + '.' + str(max_rad) +'.pdf', bbox_inches="tight")
    p_orb_scatterplot = plt.show()
    return p_orb_scatterplot

#HJ_scatter_all_color()

#HJ_scatter()
# df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#") 
# print(max(df["pl_rade"])) #### the max value is apparently 77.342 but when plotting it stops around 20


def HR_scatter(min_p_orb = 0, max_p_orb = 10, 
                       min_rad = 8, max_rad = 30, 
                       min_mass = 95, max_mass = 6356, brightness = 13.5):
    """
    Function calculating absolute magnitudes and returning a plot of the distribution
    of these versus effective stellar temperatures for a specified subset of the 
    NASA Exoplanet Archive's Planetary Systems Composite Data. 
    
    Parameters:
        - min_p_orb (float): minimum orbital period in days
        - max_p_orb (float): maximum orbital period in days
        - min_rad (float): minimum radius in Earth radii
        - max_rad (float): maximum radius in Earth radii
        - min_mass (float): minimum mass in Earth masses
        - max_mass (float): maximum mass in Earth masses

    Returns:
        - HR_scatterplot (figure object): A plot of the HJ candidates for 
        effective stellar temperature versus absolute magnitude given the specified parameters.
    """
    df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#") # accessing csvfile as dataframe
    sel = ((df["pl_orbper"]>=min_p_orb) & (df["pl_orbper"]<=max_p_orb) 
           & (df["pl_rade"]>=min_rad) & (df["pl_rade"]<=max_rad) 
           & (df["pl_bmasse"]>=min_mass) & (df["pl_bmasse"]<=max_mass)) # selection criteria
    df_sample = df[sel]
    app_mag = df_sample["sy_tmag"]
    t_eff = df_sample["st_teff"]
    dist_pc = df_sample["sy_dist"]

    # absoulte magnitude formula: M = m - 5*log10(dist_pc) + 5
    abs_mag = (app_mag - 5*np.log10(dist_pc) + 5)

    magsel = sel & (df["sy_tmag"]< brightness)
    df_mag = df[magsel]
    mag_app_mag = df_mag["sy_tmag"]
    mag_t_eff = df_mag["st_teff"]
    mag_dist_pc = df_mag["sy_dist"]

    mag_abs_mag = (mag_app_mag - 5*np.log10(mag_dist_pc) + 5)

    # plot HR
    fig, ax = plt.subplots()
    plt.grid(visible=True, which='both', axis='both')
    ax.scatter(t_eff, abs_mag, marker='.', color='k', s=7, alpha=1, label='TESS mag ≥ ' + str(brightness)) # s=size | opacity alpha=0<x<1
    ax.scatter(mag_t_eff, mag_abs_mag, marker='.', color='c', s=7, alpha=1, label='TESS mag < ' + str(brightness))
    ax.set_ylim(top=10)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel('Effective Temperature [K]', fontsize=20)
    ax.set_ylabel('Absolute magnitude [mag]', fontsize=20)
    ax.set_title("All stars hosting planet candidates with " + "\n" + str(min_mass) + " < m < " + str(max_mass) + r" $M_{Earth}$, " +
                 str(min_rad) + " < r < " + str(max_rad) + r" $r_{Earth}$, and " + 
                 str(min_p_orb) + r" < $p_{orb}$ < " + str(max_p_orb) + " days", fontsize = 12)
    plt.savefig('TESSmag_HR_scatter_color' + str(min_p_orb) + '.' + str(max_p_orb) + '.' + str(min_mass) + '.' +
                str(max_mass) + '.' + str(min_rad) + '.' + str(max_rad) + '.' + str(brightness) +'no_white_dwarf.png', bbox_inches="tight", dpi=1000)
    # HR_scatterplot = plt.show()
    # return HR_scatterplot

# HR_scatter()

# def hex_to_RGB(hex_str):
#     """ #FFFFFF -> [255,255,255]"""
#     #Pass 16 to the integer function for change of base
#     return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

# def get_color_gradient(c1, c2, n):
#     """
#     Given two hex colors, returns a color gradient
#     with n colors.
#     """
#     assert n > 1
#     c1_rgb = np.array(hex_to_RGB(c1))/255
#     c2_rgb = np.array(hex_to_RGB(c2))/255
#     mix_pcts = [x/(n-1) for x in range(n)]
#     rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
#     return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]


def HJ_brightness(min_p_orb = 0, max_p_orb = 10, 
                       min_rad = 8, max_rad = 30, 
                       min_mass = 95, max_mass = 6356, mags = 13.5):
    """
    Function returning a histogram of the distribution of TESS brightnesses for a 
    specified subset of the NASA Exoplanet Archive's Planetary Systems Composite Data.
    Black line in diagram marks cutoff magnitude.
    
    Parameters:
        - min_p_orb (float): minimum orbital period in days
        - max_p_orb (float): maximum orbital period in days
        - min_rad (float): minimum radius in Earth radii
        - max_rad (float): maximum radius in Earth radii
        - min_mass (float): minimum mass in Earth masses
        - max_mass (float): maximum mass in Earth masses
        - mags (float): maximum brightness value (inverse scale so max value
        indicates lowest brightness allowed) from TESS.

    Returns:
        - bright_hist (figure object): A histogram of the brightnesses of HJ candidates.
        As telescopic data cannot resolve objects with mag > 13.5, a black dotted line indicates
        this threshold.
    """
    df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#") # accessing csvfile as dataframe
    sel = ((df["pl_orbper"]>=min_p_orb) & (df["pl_orbper"]<=max_p_orb) 
           & (df["pl_rade"]>=min_rad) & (df["pl_rade"]<=max_rad) 
           & (df["pl_bmasse"]>=min_mass) & (df["pl_bmasse"]<=max_mass))
    df_sample = df[sel]
    # fp_orb = df_sample["pl_orbper"]
    # fr_e = df_sample["pl_rade"]
    # fm_e = df_sample["pl_bmasse"]
    fmag = df_sample["sy_tmag"]

    mag_sel = sel & (df["sy_tmag"]<mags)
    df_brightsample = df[mag_sel]

    bright_fmag = df_brightsample["sy_tmag"]
    fmag_fraction = len(bright_fmag) / len(fmag)

    # histogram
    fig, ax = plt.subplots()
    ax.hist(fmag, bins=20, color='#FD6A02', edgecolor='white', linewidth=1)
    ax.set_title("TESS Brightness distribution of HJ sample")
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Brightness [mag]")
    ax.vlines(mags, ymin=0, ymax=100, color='k', linestyle='dotted')
    ax.text(14, 60, str(len(bright_fmag)) + '/' + str(len(fmag)) + ' (' + str(round(100*fmag_fraction, 2)) + '%) objects \nbelow mag 13.5')
    bright_hist = plt.show()
    
    return bright_hist

#HJ_brightness()

def population_parameters(min_p_orb = 0, max_p_orb = 10, 
                       min_rad = 8, max_rad = 30, 
                       min_mass = 95, max_mass = 6356):
    """
    Given parameters, returns plots of orbital period vs mass, radius
    HR diagram and histogram of brightnesses.

     Parameters:
        - min_p_orb (float): minimum orbital period in days
        - max_p_orb (float): maximum orbital period in days
        - min_rad (float): minimum radius in Earth radii
        - max_rad (float): maximum radius in Earth radii
        - min_mass (float): minimum mass in Earth masses
        - max_mass (float): maximum mass in Earth masses

    Returns:
        - mosaic (figure object): matplotlib object with all three plots.    
    """
    plot1 = HJ_scatter(min_p_orb, max_p_orb, min_rad, max_rad, min_mass, max_mass)
    plot2 = HR_scatter(min_p_orb, max_p_orb, min_rad, max_rad, min_mass, max_mass)
    plot3 = HJ_brightness(min_p_orb, max_p_orb, min_rad, max_rad, min_mass, max_mass, 13.5)

    config = """
    AABB
    AACC
    """

    fig = plt.figure();axs = fig.subplot_mosaic(config)
    axs['A'].plot(plot1)
    axs['B'].plot(plot2)
    axs['C'].plot(plot3)
    plt.axis('off')
    mosaic = plt.show()
    return mosaic

#population_parameters()

def stellar_magnitude_check(star):
    """
    Gives a boolean saying true if the TESS magnitude for one star
    is brighter than TESS 13.5, and false otherwise.

    Parameters:
        - star (str): the 'TIC ID' of the star, given by the TESS Input Catalog.

    Returns:
        - boolean    
    """
    df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#")
    select_star = df['tic_id'] == star
    cell = float(df.loc[select_star, 'sy_tmag'].iloc[0]) 
    if cell < 13.5:
        return True
    else:
        return False
    

def stellar_rotation_period(star, author_input):
    """
    Function taking a star name and author and returning the stellar rotation
    period of said star.

    Parameters:
        - star (str): the TIC ID for the star. Format: "TIC 123456789"
        - author_input (str): "QLP" or "SPOC".

    Returns:
        - stellar_rot_period (float): the stellar rotation period of the star.    


    append to a list?
    """
    if not stellar_magnitude_check(star):
        print("The star " + star + " is not bright enough for a stellar rotation measurement.")
    else:
        sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
        for i in range(len(sr)):
            case = sr[i].download()
            pg = case.normalize(unit='ppm').to_periodogram(maximum_period=28)
            stellar_rot_period = pg.period_at_max_power
        return stellar_rot_period
    
#stellar_rotation_period("TIC 268301217", "QLP") #TOI-1937
# stellar_rotation_period("TIC 100100827", "QLP") #WASP-18


def plot_stellar_rotation_period(star, author_input):
    """
    Function taking a star name and author and returning a plot of the stellar rotation
    period of said star.

    Parameters:
        - star (str): the TIC ID for the star. Format: "TIC 123456789"
        - author_input (str): "QLP" or "SPOC".

    Returns:
        - diagram (figure object): the stellar rotation period plot for the star. 
    """
    if not stellar_magnitude_check(star):
        print("The star " + star + " is not bright enough for a stellar rotation measurement.")
    else:
        sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
        for i in range(len(sr)):
            case = sr[i].download()
            pg = case.normalize(unit='ppm').to_periodogram(maximum_period=28)
            period = pg.period_at_max_power
            ax = case.fold(period).scatter(label=f'Period = {period.value:.3f} d')
            ax.set_title("Folded on max power: " + star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))
            plt.savefig(star + "_" + str(sr[i].mission) + "_" + str(sr[i].exptime) + '_stellar_rotation_period.png', bbox_inches="tight", dpi=400)
        diagram = plt.show()   
        return diagram 

#plot_stellar_rotation_period("TIC 268301217", "QLP") #TOI-1937
# stellar_rotation_period_plot("TIC 100100827", "QLP") #WASP-18   

def plot_periodogram(star, author_input):
    """
    Function taking a star name and author and returning a periodogram
    (power spectrum, freq vs power) of said star.

    Parameters:
        - star (str): the TIC ID for the star. Format: "TIC 123456789"
        - author_input (str): "QLP" or "SPOC".

    Returns:    
        - periodo (figure object): plot of periodogram.
    """
    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
    for i in range(len(sr)):
        lc = sr[i].download()
        pg = lc.normalize(unit='ppm').to_periodogram(maximum_period=28)
        ax = pg.plot(view='period')
        ax.set_title("Periodogram " + star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))
        plt.savefig(star + "_" + str(sr[i].mission) + "_" + str(sr[i].exptime) + '_periodogram_period.png', bbox_inches="tight", dpi=400)
    periodo = plt.show()
    return periodo    

def plot_lightcurve(star, author_input):
    """
    Function taking a star name and author and returning a lightcurve
    (flux over time) of said star.

    Parameters:
        - star (str): the TIC ID for the star. Format: "TIC 123456789"
        - author_input (str): "QLP" or "SPOC".

    Returns:
        - lightc (figure object): lightcurve plot.
    """
    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
    for i in range(len(sr)):
        lc = sr[i].download()
        ax = lc.plot()
        ax.set_title(star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))
        plt.savefig(star + "_" + str(sr[i].mission) + "_" + str(sr[i].exptime) + '_lightcurve.png', bbox_inches="tight", dpi=400)
    lightc = plt.show()
    return lightc

#plot_periodogram("TIC 268301217", "QLP")
#plot_periodogram("TIC 100100827", "QLP")
# lightcurve("TIC 268301217", "QLP")
# lightcurve("TIC 100100827", "QLP")








def stellar_rot_vs_t_eff():
    """
    Returns:
        - plot of stelar rotation vs effective temp
    """


# stellar rotation period vs t_eff

def loop_display_all_fluxes(object, author_input):
    """
    Given an object name and mission author (QLP or SPOC), returns plots of
    pdcsap_flux/kspsap_flux, sap_flux and sap_bkg for all search results 
    matching the input variables.

    QLP restricted to exposure time 1800s, SPOC to 120s.

    Parameters:
        - object (str) - name of the star
        - author_input (str) - name of author as specified in search_lightcurve
        search results. Function only accepts "QLP" or "SPOC".

    Returns:
        - figure (figure object) - a lightcurve plot of the three fluxes
        for the specified 'object'.



    """
    sr = lk.search_lightcurve(object, author=author_input, exptime=AUTHOR_DURATION[author_input])
    for i in range(len(sr)):
        case = sr[i].download()
        ax = case.plot(column=FLUX_REFERENCE[author_input], label=FLUX_LABEL[author_input], normalize=True, alpha=0.5)
        ax1 = case.plot(column='sap_flux', label='SAP Flux', normalize=True, alpha=0.5, ax=ax)
        case.plot(column='sap_bkg', label='SAP Background', color='k', lw='1', normalize=True, ax=ax1.twinx())
        ax.set_title("All fluxes " + str(object) + str(sr[i].mission) + " " + str(sr[i].exptime))
        plt.savefig(object + "_" + str(sr[i].mission) + "_" + str(sr[i].exptime) + '_three_fluxes.png', bbox_inches="tight", dpi=400)
    figure = plt.show()
    return figure

#loop_display_all_fluxes("TIC 268301217", "QLP")

def all_four_plots_for_interim(star, author_input):
    """
    
    """
    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
    for i in range(len(sr)):
        case = sr[i].download()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

        # first one
        lc = sr[i].download()
        ax1 = lc.plot()
        ax1.set_title(star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))


        # second
        pg = lc.normalize(unit='ppm').to_periodogram()
        ax2 = pg.plot(view='period')
        ax2.set_title("Periodogram " + star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))


        # third
        axa = lc.plot(column=FLUX_REFERENCE[author_input], label=FLUX_LABEL[author_input], normalize=True, alpha=0.5)
        axb = lc.plot(column='sap_flux', label='SAP Flux', normalize=True, alpha=0.5, ax=axa)
        ax3 = lc.plot(column='sap_bkg', label='SAP Background', color='k', lw='1', normalize=True, ax=axb.twinx())
        ax3.set_title("All fluxes " + str(sr[i].mission) + " " + str(sr[i].exptime))

        #fourth 
        period = pg.period_at_max_power
        ax4 = lc.fold(period).scatter(label=f'Period = {period.value:.3f} d')
        ax4.set_title("Folded on max power: " + star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))
    mosaic = plt.show
    return mosaic

#all_four_plots_for_interim("TOI-1937", "QLP")

# def mosaic_plot_try_again():
#     """
#     """
#     image1 = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\Week3\\mosaic\\TIC 268301217_['TESS Sector 07']_[1800.] s_lightcurve.png"
#     image2 = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\Week3\\mosaic\\TIC 268301217_['TESS Sector 07']_[1800.] s_periodogram_period.png"
#     image3 = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\Week3\\mosaic\\TIC 268301217_['TESS Sector 07']_[1800.] s_three_fluxes.png"
#     image4 = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\Week3\\mosaic\\TIC 268301217_['TESS Sector 07']_[1800.] s_stellar_rotation_period.png"
#     open_image1 = np.asarray(Image.open(image1))
#     open_image2 = np.asarray(Image.open(image2))
#     open_image3 = np.asarray(Image.open(image3))
#     open_image4 = np.asarray(Image.open(image4))

#     config = """
#     AABB
#     CCDD
#     """

#     fig = plt.figure();axs = fig.subplot_mosaic(config, width_ratios=(3,3,3,3), height_ratios=(1,1))

#     axs['A'].imshow(open_image1)
#     axs['A'].axis('off')
#     axs['A'].set_title('a)')

#     axs['B'].imshow(open_image2)
#     axs['B'].axis('off')
#     axs['B'].set_title('b)')

#     axs['C'].imshow(open_image3)
#     axs['C'].axis('off')
#     axs['C'].set_title('c)')

#     axs['D'].imshow(open_image4)
#     axs['D'].axis('off')
#     axs['D'].set_title('d)')

#     fig.suptitle("Example of plots generated by the pipeline")

#     plt.savefig('mosaic_plot_TOI-1937.png', bbox_inches="tight", dpi=400)

#     mosaic = plt.show()
#     return mosaic

#mosaic_plot_try_again()

def fits_header_check(star, author_input, i):
    """
    Checks whether there is a fits header for the star
    """
    assert star.startswith("TIC ")

    ticid = star[4:]

    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])

    lc = sr[i].download()
    #print(lc.meta['SECTOR'])

    dirpath = "C:\\Users\\Elin Stenmark\\.lightkurve\\cache\\mastDownload\\*"

    # can get sector from mission in download

    fitspath = glob(join(dirpath, f"*s{str(lc.meta['SECTOR']).zfill(4)}*{ticid}*", "*lc.fits"))
    #print(file)
    # ['C:\\Users\\Elin Stenmark\\.lightkurve\\cache\\mastDownload\\TESS\\tess2018234235059-s0002-0000000100100827-0121-s\\tess2018234235059-s0002-0000000100100827-0121-s_lc.fits']   
    
    if (len(fitspath) == 0):
        return False
    else:
        return True


def get_fits_header_info(star, author_input, i):
    """
    Parameters:
        - star (str): WITH TIC in the name, i.e. "TIC 100100827".
    """
    #if not 'FILENAME'

    assert star.startswith("TIC ")

    ticid = star[4:]

    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])

    lc = sr[i].download()
    #print(lc.meta['SECTOR'])

    dirpath = "C:\\Users\\Elin Stenmark\\.lightkurve\\cache\\mastDownload\\*"

    #lc.to_fits(path=dirpath, overwrite=True)

    # can get sector from mission in download

    fitspath = glob(join(dirpath, f"*s{str(lc.meta['SECTOR']).zfill(4)}*{ticid}*", "*lc.fits"))
    #print(file)
    # ['C:\\Users\\Elin Stenmark\\.lightkurve\\cache\\mastDownload\\TESS\\tess2018234235059-s0002-0000000100100827-0121-s\\tess2018234235059-s0002-0000000100100827-0121-s_lc.fits']   
    hdul = fits.open(fitspath[0])

    #thisismetrying = "C:\\Users\\Elin Stenmark\\.lightkurve\\cache\\mastDownload\\TESS\\tess2018234235059-s0002-0000000100100827-0121-s\\tess2018234235059-s0002-0000000100100827-0121-s_lc.fits"

    # this command worked when I passed in the copied path w the double slashes, but not the "file" variable (see output of 'file' above)
    #hdul = fits.open(thisismetrying)
    #print(hdul)
    #[<astropy.io.fits.hdu.image.PrimaryHDU object at 0x000002501850EA60>, <astropy.io.fits.hdu.table.BinTableHDU object at 0x000002501860E5E0>, <astropy.io.fits.hdu.image.ImageHDU object at 0x00000250185FF250>]

    hdr2 = hdul[0].header

    # FITS header stuff
    #import IPython; IPython.embed()
    t_start = hdr2["TSTART"]
    t_stop = hdr2["TSTOP"]
    date_start = hdr2["DATE-OBS"].split("T")[0]
    date_end = hdr2["DATE-END"].split("T")[0]
    t_eff = hdr2["TEFF"]
    log_g = round(hdr2["LOGG"], 2)
    mh = hdr2["MH"]
    rad = round(hdr2["RADIUS"], 2)
    t_mag = hdr2["TESSMAG"]
    ra = round(hdr2["RA_OBJ"], 5)
    dec = round(hdr2["DEC_OBJ"], 5)
    #telescope = hdr2["TELESCOP"]
    instrument = hdr2["INSTRUME"]
    data_release = hdr2["DATA_REL"]
    obj = hdr2["OBJECT"]
    #ticid = hdr2["TICID"]
    sector = hdr2["SECTOR"]
    camera = hdr2["CAMERA"]
    ccd = hdr2["CCD"]

    text = (
        "General info:\n"
        f"Object: {obj}\n"
        f"Sector {sector}, Camera {camera}, CCD {ccd}\n"
        f"Instrument: {instrument}\n"
        f"Data release {data_release}\n"
        "--------\n"
        "Observational date:\n"
        f"Start: {date_start}\n"
        f"End:   {date_end}\n"
        "In BTJD Time:\n"
        f"Start: {t_start}\n"
        f"End:   {t_stop}\n"
        "--------\n"
        "Star parameters:\n"
        "Effective temperature: " + f"{t_eff} K\n"
        f"log g: {log_g}\n"
        f"TESS Mag: {t_mag}\n"
        f"Metallicity: {mh} log10([M/H])\n"
        f"Radius: {rad} solar\n"
        f"RA: {ra}\nDEC: {dec}"

    )

    return text, hdr2

#get_fits_header_info("TIC 100100827", "QLP", 0)

def get_fits_header_info_all_headers(star, author_input, i):
    assert star.startswith("TIC ")

    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])

    lc = sr[i].download()

    dirpath = lc.meta["FILENAME"]

    hdul = fits.open(dirpath)

    # get headers
    hdr_prim = hdul[0].header
    hdr_lc = hdul[1].header
    # construct dictionary
    hdr_primkeys = [b for b in hdr_prim.keys()]
    hdr_primvals = [c for c in hdr_prim.values()]
    hdr_primdict = {b:c for b,c in zip(hdr_primkeys, hdr_primvals)}

    hdr_lckeys = [d for d in hdr_lc.keys()]
    hdr_lcvals = [e for e in hdr_lc.values()]
    hdr_lcdict = {d:e for d,e in zip(hdr_lckeys, hdr_lcvals)}

    hdr_mergerdict = deepcopy(hdr_primdict)
    hdr_mergerdict.update(hdr_lcdict)

    df_hdr_merged = pd.DataFrame(hdr_mergerdict, index=[0])
    return df_hdr_merged

# print(get_fits_header_info_all_headers("TIC 148563075", "QLP", 0))
# info = get_fits_header_info_all_headers("TIC 148563075", "QLP", 0)
# print("hey look here")
# print(info["RADIUS"].values[0])

def format_fits_header_info_all_headers(star, author_input, i):
    info = get_fits_header_info_all_headers(star, author_input, i)

    t_start = round(info["TSTART"].values[0], 2) if "TSTART" in info else ""
    t_stop = round(info["TSTOP"].values[0], 2) if "TSTOP" in info else ""
    date_start = info["DATE-OBS"].values[0].split("T")[0] if "DATE-OBS" in info else ""
    date_end = info["DATE-END"].values[0].split("T")[0] if "DATE-END" in info else ""
    t_eff = info["TEFF"].values[0] if "TEFF" in info else ""
    log_g = round(info["LOGG"].values[0], 2) if "LOGG" in info else ""
    mh = round(info["MH"].values[0], 2) if "MH" in info else ""
    rad = round(info["RADIUS"].values[0], 2) if "RADIUS" in info else ""
    t_mag = round(info["TESSMAG"].values[0], 2) if "TESSMAG" in info else ""
    ra = round(info["RA_OBJ"].values[0], 5) if "RA_OBJ" in info else ""
    dec = round(info["DEC_OBJ"].values[0], 5) if "DEC_OBJ" in info else ""
    #telescope = hdr2["TELESCOP"]
    instrument = info["INSTRUME"].values[0] if "INSTRUME" in info else ""
    data_release = info["DATA_REL"].values[0] if "DATA_REL" in info else ""
    obj = info["OBJECT"].values[0] if "OBJECT" in info else ""
    #ticid = hdr2["TICID"]
    sector = info["SECTOR"].values[0] if "SECTOR" in info else ""
    camera = info["CAMERA"].values[0] if "CAMERA" in info else ""
    ccd = info["CCD"].values[0] if "CCD" in info else ""

    text = (
        "General info:\n"
        f"Object: {obj}\n"
        f"Sector {sector}, Camera {camera}, CCD {ccd}\n"
        f"Instrument: {instrument}\n"
        f"Data release {data_release}\n"
        "--------\n"
        "Observational date:\n"
        f"Start: {date_start}\n"
        f"End:   {date_end}\n"
        "In BTJD Time:\n"
        f"Start: {t_start}\n"
        f"End:   {t_stop}\n"
        "--------\n"
        "Star parameters:\n"
        "Effective temperature: " + f"{t_eff} K\n"
        f"log g: {log_g}\n"
        f"TESS Mag: {t_mag}\n"
        f"Metallicity: {mh}\n"
        f"Radius: {rad} solar\n"
        f"RA: {ra}\nDEC: {dec}"

    )
    return text





def vetting_plot_stellar_rotation_period(star, author_input, i):
    """
    Function taking a star name and author and returning the stellar rotation
    period of said star.

    Parameters:
        - star (str): the TIC ID for the star. Format: "TIC 123456789"
        - author_input (str): "QLP" or "SPOC".

    Returns:
        - stellar_rot_period (Quantity): the stellar rotation period of the star.    

    """
    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
    case = sr[i].download()
    sel = get_sel(star, case)
    pg = case[sel].normalize(unit='ppm').to_periodogram(maximum_period=28)
    stellar_rot_period = pg.period_at_max_power
    return stellar_rot_period

def get_ra_dec(star, author_input, i):
    """
    """
    assert star.startswith("TIC ")

    ticid = star[4:]

    sr = lk.search_lightcurve(star, author=author_input)

    lc = sr[i].download()

    #dirpath = "C:\\Users\\Elin Stenmark\\.lightkurve\\cache\\mastDownload\\TESS\\"
    dirpath = lc.meta["FILENAME"]

    #fitspath = glob(join(dirpath, f"*s{str(lc.meta['SECTOR']).zfill(4)}*{ticid}*", "*lc.fits")) 
    hdul = fits.open(dirpath[0])
    hdr2 = hdul[0].header

    ra = round(hdr2["RA_OBJ"], 5)
    dec = round(hdr2["DEC_OBJ"], 5) 
    return ra, dec

def alt_get_ra_dec(star, author_input, i):
    """
    """
    info = get_fits_header_info_all_headers(star, author_input, i)
    ra = round(info["RA_OBJ"], 5)
    dec = round(info["DEC_OBJ"], 5)
    return ra, dec

#alt_get_ra_dec("TIC 100100827", "QLP", 0)

def get_stars(min_p_orb = 0, max_p_orb = 10, 
                       min_rad = 8, max_rad = 30, 
                       min_mass = 95, max_mass = 6356, ):
    """
    """
    df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#")
    sel = ((df["pl_orbper"]>=min_p_orb) & (df["pl_orbper"]<=max_p_orb) 
           & (df["pl_rade"]>=min_rad) & (df["pl_rade"]<=max_rad) 
           & (df["pl_bmasse"]>=min_mass) & (df["pl_bmasse"]<=max_mass))
    df_sample = df[sel]
    ticid_list = df_sample["tic_id"]
    newlist = ticid_list.to_list()

    newlist = [str(item) for item in newlist]
    test_subset = newlist[275:290]
    # for i in newlist:
    #     newlist[i] = str(newlist[i])

    return test_subset
    # how do I access specific row in column/iterate over?

# print(get_stars())

def get_pipeline_dict_values(star, author_input, i):
    """
    Function taking a star name and author and returning the stellar rotation
    period of said star.

    Parameters:
        - star (str): the TIC ID for the star. Format: "TIC 123456789"
        - author_input (str): "QLP" or "SPOC".

    Returns:
        - stellar_rot_period (Quantity): the stellar rotation period of the star.    

    """
    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
    case = sr[i].download()
    sel = get_sel(star, case)
    pg = case[sel].normalize(unit='ppm').to_periodogram(maximum_period=28)
    stellar_rot_period = pg.period_at_max_power
    max_power = pg.max_power
    return stellar_rot_period, max_power, sr


# def find_npeaks_periodogram(star, author_input, i):
#     """
#     """
#     sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
#     lc = sr[i].download()
#     times = lc.time
#     fluxes = lc.flux
#     errs = lc.flux_err
#     # pgdict = pgen_lsp(times, fluxes, errs, magsarefluxes=True)
#     # return print(pgdict)

# find_npeaks_periodogram("TIC 100100827", "QLP", 0)




def find_peaks(lspvals, lspperiods, nbestpeaks=10, period_eps=0.1):
    """
    """
    sorted_lspinds = np.argsort(lspvals)[::-1]
    sorted_lspperiods = lspperiods[sorted_lspinds]
    sorted_lspvals = lspvals[sorted_lspinds]

    nbestperiods = [sorted_lspperiods[0]]
    nbestvals = [sorted_lspvals[0]]
    peakcount = 1
    prevperiod = sorted_lspperiods[0]
    # IPython.embed()
    for lspperiod, lspval in zip(sorted_lspperiods, sorted_lspvals):
        if peakcount == nbestpeaks:
            break
        perioddiff = abs(lspperiod - prevperiod)
        bestperiodsdiff = [abs(lspperiod - x) for x in nbestperiods]
    
        if (perioddiff > (period_eps*prevperiod) and
                np.all([x > (period_eps*lspperiod) for x in bestperiodsdiff])):

                nbestperiods.append(lspperiod)
                nbestvals.append(lspval)
                peakcount = peakcount + 1

        prevperiod = lspperiod
    return nbestperiods, nbestvals 


def peakfinder(star, author_input, i):
    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
    lc = sr[i].download()
    sel = get_sel(star, lc)
    pg = lc[sel].normalize(unit='ppm').to_periodogram(maximum_period=28)
    # IPython.embed()
    lspvals = pg.power.value
    lspperiods = pg.period.value
    sel2 = np.isfinite(lspvals) & np.isfinite(lspperiods)
    nbestperiods, nbestvals = find_peaks(lspvals[sel2], lspperiods[sel2])
    points = (list(zip(nbestperiods, nbestvals)))
    return points

# what = peakfinder("TIC 100100827", "QLP", 0)[0]
# print(what)
# print(what[0])
# print(what[1])

# tehe = get_pipeline_dict_values("TIC 100100827", "QLP", 0)
# print(tehe)

def to_single_row_df(star, author_input, i):
    """
    """
    hdr2 = get_fits_header_info_all_headers(star, author_input, i)

    hdr2keys = [b for b in hdr2.keys()]
    hdr2vals = [c for c in hdr2.values[0]]
    
    colnames = ("TSTART TSTOP DATE-OBS DATE-END TEFF LOGG MH RADIUS TESSMAG RA_OBJ DEC_OBJ INSTRUME DATA_REL OBJECT SECTOR CAMERA CCD").split()
    hdr2dict = {b:c for b,c in zip(hdr2keys, hdr2vals) if b in colnames}
    # df_fits = pd.DataFrame(hdr2dict, index=[0])
    # df_fits.to_csv()

    df_nea = pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#")
    select_star = (df_nea['tic_id'] == star)
    orb_per_cell = df_nea.loc[select_star, 'pl_orbper'].iloc[0]
    p_rot_cell = df_nea.loc[select_star, 'st_rotp'].iloc[0]
    dist_cell = df_nea.loc[select_star, 'sy_dist'].iloc[0]

    nea_dict = {"nea_p_orb": orb_per_cell, "nea_p_rot": p_rot_cell, "nea_dist": dist_cell}
    #x = print(deepcopy(hdr2dict))

    merged_dict = deepcopy(hdr2dict)
    merged_dict.update(nea_dict)

    pipe_p_rot, pipe_max_pow, sr = get_pipeline_dict_values(star, author_input, i)

    p_rot_numbers_only = str(pipe_p_rot).split(" ")[0]
    new_stel_rot = round(float(p_rot_numbers_only), 2)

    max_pow_numbers_only = str(pipe_max_pow).split(" ")[0]
    new_max_pow = round(float(max_pow_numbers_only), 2)


    pipeline_dict = {"pipe_p_rot": new_stel_rot, "pipe_max_power": new_max_pow, "pipe_author": author_input}

    merged_dict.update(pipeline_dict)

    points = peakfinder(star, author_input, i)
    power_peak_power_dict = {"A_power": points[0][1], "B_power": points[1][1], "C_power": points[2][1], "D_power": points[3][1], "E_power": points[4][1], "F_power": points[5][1]}
    power_peak_period_dict = {"A_period": points[0][0], "B_period": points[1][0], "C_period": points[2][0], "D_period": points[3][0], "E_period": points[4][0], "F_period": points[5][0]}

    merged_dict.update(power_peak_power_dict)
    merged_dict.update(power_peak_period_dict)

    merged_df = pd.DataFrame(merged_dict, index=[0])

    return merged_df

# to_single_row_df("TIC 100100827", "QLP", 0)

# points = peakfinder("TIC 100100827", "QLP", 0)
# power_peak_dict = {"A": points[0], "B": points[1], "C": points[2], "D": points[3], "E": points[4], "F": points[5]}

# print(power_peak_dict)
        # df.insert(loc=1, column='ticid', value=obj)
        # df.insert(loc=2, column='sector', value=sector)
        # df.insert(loc=3, column='camera', value=camera)
        # df.insert(loc=4, column='CCD', value=ccd)
        # df.insert(loc=5, column='instrument', value=instrument)
        # df.insert(loc=6, column='data_release', value=data_release)
        # df.insert(loc=7, column='start_time', value=t_start)
        # df.insert(loc=8, column='end_time', value=t_stop)
        # df.insert(loc=9, column='start_date', value=date_start)
        # df.insert(loc=10, column='end_date', value=date_end)

#to_single_row_df("TIC 100100827", "QLP", 0)

# def test_to_see_if_df_fits_worked(star, author_input, i):
#     """
#     """
#     text, hdr2 = get_fits_header_info(star, author_input, i)
#     hdr2keys = [b for b in hdr2.keys()]
#     hdr2vals = [c for c in hdr2.values()]
    
#     colnames = ("TSTART TSTOP DATE-OBS DATE-END TEFF LOGG MH RADIUS TESSMAG RA_OBJ DEC_OBJ INSTRUME DATA_REL OBJECT SECTOR CAMERA CCD").split()
#     hdr2dict = {b:c for b,c in zip(hdr2keys, hdr2vals) if b in colnames}
#     df = pd.DataFrame(hdr2dict, index=[0])
#     return print(df)

# test_to_see_if_df_fits_worked("TIC 100100827", "QLP", 0)

def write_df_to_csv(star, author_input, i, version):
    hdr2 = get_fits_header_info_all_headers(star, author_input, i)
    sector = hdr2["SECTOR"].values[0]
    nan1, nan2, sr = get_pipeline_dict_values(star, author_input, i)
    exptime = str(sr[i].exptime).split(" ")[0]

    outdir = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\SURF_code\\results\\csv_info_files\\" + f"{version}\\"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        filename = outdir + star + "_" + author_input + "_" + str(sector) + "_" + str(exptime) + f"_{version}.csv"
        file = open(filename, 'w')
        dataframe = to_single_row_df(star, author_input, i)
        dataframe.to_csv(path_or_buf=file)
    else:    
        filename = outdir + star + "_" + author_input + "_" + str(sector) + "_" + str(exptime) + f"_{version}.csv"
        file = open(filename, 'w')
        dataframe = to_single_row_df(star, author_input, i)
        dataframe.to_csv(path_or_buf=file)

#write_df_to_csv("TIC 100100827", "QLP", 0)
# 
# 

def get_nea_transit_mask_params(star):
    """
    """
    df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#")
    select_star = (df['tic_id'] == star)
    cell_tranmid = float(df.loc[select_star, 'pl_tranmid'].iloc[0])
    cell_tranmid = (cell_tranmid - 2457000)
    cell_orbper = float(df.loc[select_star, 'pl_orbper'].iloc[0])
    cell_trandur = float(df.loc[select_star, 'pl_trandur'].iloc[0])
    cell_trandur = cell_trandur/24
    # import IPython; IPython.embed()
    return cell_tranmid, cell_orbper, cell_trandur

#print(get_nea_transit_mask_params("TIC 63189173"))
# print(get_nea_transit_mask_params("TIC 402026209"))
    
def get_sel(star, lc):
    """
    """
    tranmid, orbper, trandur = get_nea_transit_mask_params(star)
    mask = lc.create_transit_mask(orbper, tranmid, trandur)
    sel = (lc['quality'] == 0) & ~(mask)
    return sel



def plot_vetting_mosaic(star, author_input, i, sr, version):
    """
    """
    config = (
    """
    AA.BBEE
    CC.DDFF
    """)

    fig, axs = plt.subplot_mosaic(config, figsize=(20,15), width_ratios=(1,1,0.2,1,1,0.6, 0.6))

    # lc
    #ax = axs['A']
    lc = sr[i].download()
    sel = get_sel(star, lc)

    # axs['A'] = lc.plot()
    lc[sel].plot(ax=axs['A'])
    #axs['A'].plot(lc.time.value, lc.flux.value)
    axs['A'].set_title(star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))
    

    # pg
    pg = lc[sel].normalize(unit='ppm').to_periodogram(maximum_period=28)
    # axs['B'] = pg.plot(view='period')
    pg.plot(view='period', scale='log', ax=axs['B'])
    axs['B'].set_title("Periodogram " + star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))
    axs['B'].set_xlim(left=0, right=28)

    color1 = '#01013F'
    color2 = '#0062B1'
    color3 = '#51A2D5'
    color4 = '#F48BA9'
    color5 = '#FF6EC7'
    color6 = '#FF3B9B'

    points = peakfinder(star, author_input, i)
    exx = points[0][0]
    why = points[0][1]
    axs['B'].plot(exx, why, color=color1, marker='$A$')
    exx2 = points[1][0]
    why2 = points[1][1]
    axs['B'].plot(exx2, why2, color=color2, marker='$B$')
    exx3 = points[2][0]
    why3 = points[2][1]
    axs['B'].plot(exx3, why3, color=color3, marker='$C$')
    exx4 = points[3][0]
    why4 = points[3][1]
    axs['B'].plot(exx4, why4, color=color4, marker='$D$')
    exx5 = points[4][0]
    why5 = points[4][1]
    axs['B'].plot(exx5, why5, color=color5, marker='$E$')
    exx6 = points[5][0]
    why6 = points[5][1]
    axs['B'].plot(exx6, why6, color=color6, marker='$F$')

    

    # flux
    #ax = axs['C']
    case = sr[i].download()
    sel = get_sel(star, case)
    case[sel].plot(column=FLUX_REFERENCE[author_input], label=FLUX_LABEL[author_input], normalize=True, alpha=0.5, ax=axs['C'])
    ax1 = case[sel].plot(column='sap_flux', label='SAP Flux', normalize=True, alpha=0.5, ax=axs['C'])
    case[sel].plot(column='sap_bkg', label='SAP Background', color='k', lw='1', normalize=True, ax=ax1.twinx())
    axs['C'].set_title("All fluxes " + str(star) + str(sr[i].mission) + " " + str(sr[i].exptime))
    
    # case = sr[i].download()
    # axs['C'] = case.plot(column=FLUX_REFERENCE[author_input], label=FLUX_LABEL[author_input], normalize=True, alpha=0.5)
    # ax1 = case.plot(column='sap_flux', label='SAP Flux', normalize=True, alpha=0.5, ax=axs['C'])
    # case.plot(column='sap_bkg', label='SAP Background', color='k', lw='1', normalize=True, ax=ax1.twinx())
    # axs['C'].set_title("All fluxes " + str(star) + str(sr[i].mission) + " " + str(sr[i].exptime))
    
    # p_rot
    #ax = axs['D']
    #case = sr[i].download()
    pg = case[sel].normalize(unit='ppm').to_periodogram(maximum_period=28)
    period = pg.period_at_max_power
    case[sel].fold(period).scatter(label=f'Period = {period.value:.3f} d', ax=axs['D'])
    axs['D'].set_title("Folded on max power: " + star + " " + str(sr[i].mission) + " " + str(sr[i].exptime))

    # info text
    #case1 = sr[i].download()
    text = format_fits_header_info_all_headers(star, author_input, i)


    pipe_p_rot, pipe_max_pow, searchr = get_pipeline_dict_values(star, author_input, i)

    max_pow_numbers_only = str(pipe_max_pow).split(" ")[0]
    new_max_pow = round(float(max_pow_numbers_only), 2)

    nea_plus_merged_dict = to_single_row_df(star, author_input, i)
    nea_distance = round(nea_plus_merged_dict['nea_dist'].values[0], 1) if 'nea_dist' in nea_plus_merged_dict else ""
    nea_p_rot = round(nea_plus_merged_dict['nea_p_rot'].values[0], 2) if 'nea_p_rot' in nea_plus_merged_dict else ""
    nea_p_orb = round(nea_plus_merged_dict['nea_p_orb'].values[0], 2) if 'nea_p_orb' in nea_plus_merged_dict else ""

    axs['E'].text(0.0, 0.15, f"{text}" + "\n" + f"Max power: {new_max_pow} ppm\nDistance: {nea_distance} pc\nNEA p_rot: {nea_p_rot} d\nNEA p_orb: {nea_p_orb} d", fontsize=15)
    axs['E'].axis('off')
    axs['E'].grid('off')

    # p_rot
    stel_rot = str(vetting_plot_stellar_rotation_period(star, author_input, i))
    numbers_only = stel_rot.split(" ")[0]
    new_stel_rot = round(float(numbers_only), 2)


    axs['E'].text(0.0, 0.0, f"{author_input}\n{new_stel_rot} d", fontsize=25, weight='bold')
    axs['E'].text(0.0, -0.05, f"Peak A: {round(exx, 1)} d, {round(why, 1)} ppm", color=color1, fontsize=15)
    axs['E'].text(0.0, -0.10, f"Peak B: {round(exx2, 1)} d, {round(why2, 1)} ppm", color=color2, fontsize=15)
    axs['E'].text(0.0, -0.15, f"Peak C: {round(exx3, 1)} d, {round(why3, 1)} ppm", color=color3, fontsize=15)
    axs['E'].text(0.0, -0.20, f"Peak D: {round(exx4, 1)} d, {round(why4, 1)} ppm", color=color4, fontsize=15)
    axs['E'].text(0.0, -0.25, f"Peak E: {round(exx5, 1)} d, {round(why5, 1)} ppm", color=color5, fontsize=15)
    axs['E'].text(0.0, -0.30, f"Peak F: {round(exx6, 1)} d, {round(why6, 1)} ppm", color=color6, fontsize=15)

    # # picture
    # ra = hdr['RA_OBJ']
    # dec = hdr['DEC_OBJ']

    ra, dec = alt_get_ra_dec(star, author_input, i)

    # dss_overlay(fig, axd, ra, dec)
    current_dir = os.getcwd()

    dss, dss_hdr = skyview_stamp(ra, dec, survey='DSS2 Red',
                                scaling='Linear', convolvewith=None,
                                sizepix=220, flip=False,
                                cachedir=current_dir,
                                verbose=True, savewcsheader=True)

    
    ss = axs['F'].get_subplotspec()
    axs['F'].remove()
    axs['F'] = fig.add_subplot(ss, projection=WCS(dss_hdr))
    cset = axs['F'].imshow(dss, origin='lower', cmap=plt.cm.gray_r, zorder=-2)
    axs['F'].set_xlabel(' ')
    axs['F'].set_ylabel(' ')
    axs['F'].grid(ls='--', alpha=0.5)


    #DSS is ~1 arcsecond per pixel. overplot 1px and 2px apertures
    for ix, radius_px in enumerate([21, 21*2]):
        circle = plt.Circle(
            (220/2, 220/2), radius_px, color='C{}'.format(ix),
            fill=False, zorder=5+ix, lw=0.5, alpha=0.5
        )
        axs['F'].add_artist(circle)

    props = dict(boxstyle='square', facecolor='lightgray', alpha=0.3, pad=0.15,
                    linewidth=0)
    axs['F'].text(0.97, 0.03, 'r={1,2}px', transform=axs['F'].transAxes,
            ha='right',va='bottom', color='k', bbox=props, fontsize='x-small')

    fig.suptitle("Vetting plot " + star + " " + str(sr[i].mission) + " " + str(sr[i].exptime), fontsize=25)
    fig.tight_layout()
    savedir = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\SURF_code\\results\\vetting_plots\\" + f"{version}\\"
    nan1, nan2, sr = get_pipeline_dict_values(star, author_input, i)
    exptime = str(sr[i].exptime).split(" ")[0]
    if not os.path.exists(savedir):
        os.mkdir(savedir)
        hdr5 = get_fits_header_info_all_headers(star, author_input, i)
        sector = hdr5["SECTOR"].values[0]
        fig.savefig(savedir + star + "_" + author_input + "_" + str(sector) + "_" + str(exptime) + f"_{version}.png", bbox_inches='tight')
    else:    
        hdr5 = get_fits_header_info_all_headers(star, author_input, i)
        sector = hdr5["SECTOR"].values[0]
        fig.savefig(savedir + star + "_" + author_input + "_" + str(sector) + "_" + str(exptime) + f"_{version}.png", bbox_inches='tight')

#mosaic_third_times_the_charm("TIC 268301217", "QLP")
#plot_vetting_mosaic("TIC 100100827", "QLP")

#sr = lk.search_lightcurve("TIC 61230756")
#print(sr)

# find way to 

#print(get_stars()[3])

# def main():
#     """
#     """
#     for star in get_stars():
#         if not stellar_magnitude_check(star):
#             return 0
#         #stellar_rotation_period(star, "QLP")
#         plot_vetting_mosaic(star, "QLP")
#         sr = lk.search_lightcurve(star, author="QLP", exptime=1800)
#         for i in range(len(sr)):
#             write_df_to_csv(star, "QLP", i)
#     return
    
#main()    

# print(get_stars())

# for star in get_stars():
#     sr = lk.search_lightcurve(star, author="QLP", exptime=1800)
#     print(star)
#     print(sr)

# print(get_stars().index('TIC 341694238'))

def path_already_exists_check():
    """
    """
    # if I alr have a plot of this star don't do it again

def merge_dfs(version):
    """
    """
    csv_path = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\SURF_code\\results\\csv_info_files\\" + f"{version}\\"


def get_other_star_ids(min_p_orb = 0, max_p_orb = 10, 
                       min_rad = 8, max_rad = 30, 
                       min_mass = 95, max_mass = 6356):
    """
    """
    df=pd.read_csv("PSCompPars_2023.07.06_09.24.01.csv", comment="#")
    sel = ((df["pl_orbper"]>=min_p_orb) & (df["pl_orbper"]<=max_p_orb) 
           & (df["pl_rade"]>=min_rad) & (df["pl_rade"]<=max_rad) 
           & (df["pl_bmasse"]>=min_mass) & (df["pl_bmasse"]<=max_mass))    
    df_sel = df[sel]
    new_df_sel = df_sel[["tic_id", "gaia_id", "hostname", "hd_name", "hip_name"]].copy()
    # ticid = df_sel["tic_id"]
    # gaiaid = df_sel["gaia_id"]
    # hostname = df_sel["hostname"]
    # hdname = df_sel["hd_name"]
    # hipname = df_sel["hip_name"]
    pathname = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\Week3\\"
    filename = pathname + "starnames.csv"
    file = open(filename, 'w')
    new_df_sel.to_csv(path_or_buf=file)
    #return new_df_sel

#print(get_other_star_ids())
# get_other_star_ids()


def main2(author_input, version):
    """
    """
    start = datetime.now()
    passed_count = 0
    total_count = 0
    nan_count = 0
    dim_count = 0
    no_lc_count = 0
    empty_lc_count = 0
    individual_lc_count = 0
    path = "C:\\Users\\Elin Stenmark\\Documents\\School\\Caltech\\2022-23\\SURF\\Coding\\SURF_code\\results\\mag_check\\"
    with open(path + author_input + "_" + version + "_mag_check.txt", 'w') as f:
        f.write("# New Run:\n# -1: Star has no TICID.\n# 0: Too dim\n# 1: No lightcurve\n# 2: Plots and writes to CSV\n")
        for star in get_stars():
            total_count += 1
            if (star == 'nan'):
                f.write(f"{star}, -1\n")
                nan_count += 1
                print("-1: Star name nan - will be skipped.")
            else:    
                print(f"Progress: {passed_count}/{total_count}/{len(get_stars())}.")
                if not stellar_magnitude_check(star):
                    dim_count += 1
                    f.write(f"{star}, 0\n")
                    print(f"0: {star} not bright enough for measurement.")
                else:
                    print(f"{star} passed 0.")
                    sr = lk.search_lightcurve(star, author=author_input, exptime=AUTHOR_DURATION[author_input])
                    print(sr)
                    if (len(sr) < 1):
                        no_lc_count += 1
                        f.write(f"{star}, 1\n")
                        print(f"1: No search results found for {star}.")
                    else:
                        passed_count += 1
                        print(f"{star} passed 1.")
                        for i in range(len(sr)):
                            lc = sr[i].download()
                            sel = get_sel(star, lc)
                            if (np.all(np.isnan(lc[sel].flux))):
                                empty_lc_count += 1
                                f.write(f"{star}, 2, lightcurve {i + 1}/{len(sr)} contains no data\n")
                                print(f"2: {star} lightcurve {i + 1}/{len(sr)} contains no data.")
                            else:    
                                plot_vetting_mosaic(star, author_input, i, sr, version)
                                write_df_to_csv(star, author_input, i, version)
                                individual_lc_count += 1
                                f.write(f"{star}, 3, {i + 1}/{len(sr)}\n")
                                print(f"3: {star} lightcurve {i + 1}/{len(sr)} passed pipeline.")
        print(f"Runtime: {datetime.now() - start}")
        f.write("# " + str(passed_count) + "/" + str(total_count) + " passed pipeline. Of which, " + str(nan_count) + " are nan's.\n")
        f.write(f"# Star data:\n# -1: {nan_count}\n# 0: {dim_count}\n# 1: {no_lc_count}\n# 2: {empty_lc_count}\n# 3: {passed_count}\n")
        f.write(f"# Total lc's processed: {individual_lc_count}\n")
        f.write(f"# Runtime: {datetime.now() - start}")                
        print("Final standings: " + str(passed_count) + " passed out of " + str(total_count) + " stars, of which " + str(nan_count) + " are nan's, " + str(dim_count) + " were too dim, and " + str(no_lc_count) + " had no lc, " + str(empty_lc_count) + " individual lc's with no data. " + str(individual_lc_count) + " total lightcurves processed.")                  

#main2("QLP", "trying_for_paths")

# sr = lk.search_lightcurve("TIC 36352297", author="QLP", exptime=1800)
# plot_vetting_mosaic("TIC 36352297", "QLP", 0, sr)

# if not fits_header_check(star, author_input, i):
#                             f.write(f"{star} 2\n")
#                             print(f"2: {star} doesn't have a FITS file.")
#                         else:

# for i in range(3):
#     print("hi!")
#     e = 5*i
#     if (i==1):
#         print("wait what?")
#     else:    
#         for k in range(5):
#             o = e*k
#             print(o)

# for star in get_stars():
#     print(star)
#     if not stellar_magnitude_check(star):
#         print("not bright enough")
#     else:
#         print("shiny!")
#         sr = lk.search_lightcurve(star, author="QLP", exptime=AUTHOR_DURATION["QLP"])
#         print(sr)    



#get_fits_header_info("100100827", "QLP")
#get_fits_header_info("TIC 100100827", "QLP")    

# plt.axis('off')
# # plt.imshow(open_image)
# # plt.imshow(open_image2)
# # plt.imshow(open_image3)
# plt.show()

# if __name__ == "__main__":
#     main()
