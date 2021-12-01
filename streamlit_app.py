# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 19:32:29 2021

@author: Eoin
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from mim import MIM

st.set_page_config(layout="wide")


pi = 3.14159265
c = 299792458
eps_0 = 8.8541878128 * 10**-12
mu_0 = 4 * pi * 10**-7
e = 1.602176634 * 10**-19
hbar = 1.0545718 * 10**-34

eps_inf = 6
omega_p_1 = 5.3700 * 10**15
omega_p_2 = 2.2636 * 10**15
omega_0_1 = 0.000 * 10**15
omega_0_2 = 4.5720 * 10**15
gamma_1 = 6.216 * 10**13
gamma_2 = 1.332 * 10**15


def wl_to_omega(wl):
    return 2*pi*299_792_458/(wl*1e-9)

# def omega_to_wl(omega):
#     return wl_to_omega(omega)

num = 4.135667516e-15*299_792_458/1e-9
def wl_to_ev(wl):
    return num/wl

def ev_to_wl(ev):
    return num/ev   

def ev_to_omega(ev):
    return wl_to_omega(ev_to_wl(ev))

# def omega_to_ev(wl):
#     return omega_to_ev(wl_to_omega(wl))

def lor(omega, omega_p, omega_0, gamma):
    return omega_p**2 / (omega**2 - omega_0**2 + 1j * omega * gamma)


def eps_m(omega):
    lor1 = lor(omega, omega_p_1, omega_0_1, gamma_1)
    lor2 = lor(omega, omega_p_2, omega_0_2, gamma_2)
    return eps_inf * (1 - lor1 - lor2)


def scaled_S_lm(wls, center_wl, eff):
    omegas, center_omega_real = map(wl_to_omega, (wls, center_wl))

    center_omega_imag = ev_to_omega(
        MIM(wl_to_ev(center_wl), n, t) / (1 - eff))  # eV - = Gamma/2
    center = (center_omega_real + 1j * center_omega_imag)

    return  abs(eff * S_lm(omegas, center))**2 / 155_000


def S_lm(omega, omega_lm):
    delta_eps = eps_m(omega_lm) - 1
    return (1 - eps_inf - (delta_eps * omega_lm / (omega - omega_lm)))


def file_to_mode_name(file):
    return file.stem.replace('=', '_').split('_')[1]


def func_maker(args, body_lines, return_value):
    ldict = {}
    defline = f"def func({', '.join(args)}):\n\t"
    body = '\n\t'.join(body_lines)
    returnline = '\treturn ' + return_value
    exec(''.join((defline, body, returnline)), globals(), ldict)
    return ldict['func']


def real_factory(s_expression, parsed_txt):
    return func_maker(('f', 'D', 't', 'n'), [s_expression], parsed_txt)


def imag_factory(parsed_txt):
    func = func_maker(('real', 'D'), [], parsed_txt)

    def inner_func(real, D):
        real = wl_to_ev(real)
        out = func(real, D)
        return 0.00001 if out <= 0 else out  # prevent /0 in Lorentz

    return inner_func


def lines_factory(real_eq, imag_eq):
    def inner_func(wl):
        real = real_eq(f, D, t, n)
        efficiency = imag_eq(real, D)
        return scaled_S_lm(wl, real, efficiency)

    return inner_func


def annotate_factory(real_eq, imag_eq):
    def inner_func():
        real = real_eq(f, D, t, n)
        efficiency = imag_eq(real, D)
        return real, efficiency

    return inner_func

@st.cache(hash_funcs={type(complex(1,1)): lambda _: None})
def make_modes(folder):
    modes = defaultdict(dict)

    for file in (folder / 'real equations').iterdir():
        mode = file_to_mode_name(file)

        with open(file, 'r') as eq_file:
            s_expression = eq_file.readline()
            parsed_txt = ''.join(eq_file.read().splitlines())
            modes[f'{mode} mode']['real'] = real_factory(
                s_expression, parsed_txt)

    for file in (folder / 'imag equations').iterdir():
        mode = file_to_mode_name(file)
        with open(file, 'r') as eq_file:
            parsed_txt = ''.join(eq_file.read().splitlines())
            modes[f'{mode} mode']['imag'] = imag_factory(parsed_txt)

    for mode in modes.values():
        mode['line'] = lines_factory(mode['real'], mode['imag'])
        mode['annotate'] = annotate_factory(mode['real'], mode['imag'])

    def xlim_func():
        reals = [mode['real'](f, D, t, n) for mode in modes.values()]
        return min(reals) * 0.93, max(reals) * 1.08

    return modes, xlim_func


labels = set()


def plot_modes(modes, geometry, resolution=300, coords={}, label=False, xs=[]):
    ys = np.empty((len(modes), resolution))
    for i, (name, mode) in enumerate(modes.items()):
        y = mode['line'](xs)
        ys[i] = y
        # wl, eff = mode['annotate']()
        _label = f'{name}'  #', wl={round(wl)}nm, efficiency={np.around(eff, 2)}'

        fig.add_trace(
            go.Scatter(x=xs,
                       y=y,
                       name=_label,
                       showlegend=(not (_label in labels)),
                       mode='lines',
                       line=dict(color=colors[name], )), **coords)
        labels.add(_label)

    fig.add_trace(
        go.Scatter(x=xs,
                   y=ys.sum(axis=0),
                   name='sum',
                   showlegend=label,
                   line=dict(
                       color='white',
                       dash='dash',
                   )), **coords)
    fig.update_layout(
        height=600,
        width=800)


'''__Qausi-Normal modes of Nanoparticle on mirror__'''
plot_container = st.container()
slider_container = st.container()

extended_range = st.checkbox('allow parameters outside simulated range')
if extended_range:
    slider_args = (
        ('Facet', 0.0, 0.7, 0.3),
        ('Diameter (nm)', 20., 150., 80.),
        ('gap thickness (nm)', 0.5, 12., 1.),
        ('gap refractive index', 0.75, 2.5, 1.5),
    )
else:
    slider_args = (
        ('Facet', 0.15, 0.6, 0.3),
        ('Diameter (nm)', 40., 100., 80.),
        ('gap thickness (nm)', 0.75, 6., 1.),
        ('gap refractive index', 1., 2., 1.5),
    )

with slider_container:
    for col, param, args in zip(st.columns(4), 'fDtn', slider_args):
        with col:
            vars()[param] = st.slider(*args)

adjectives = {
    'circle': 'circular',
    'square': 'square',
    'triangle': 'triangular'
}
modes = [m + ' mode' for m in '10 11 20 21 22 2-2 33'.split()]
colors = {m: c for m, c in zip(modes, px.colors.qualitative.Plotly)}

folders = [Path('geometries') / g for g in (
    'circle',
    'triangle',
    'square',
)]

with plot_container:
    fig = make_subplots(
        rows=len(folders),
        cols=1,
        shared_xaxes=True,
        x_title='wavelength (nm)',
        y_title='',
        
    )

    xs = None
    geometries = {}
    x_lims = set()
    for folder in folders:
        geometries[folder.stem], xlim_func = make_modes(folder)
        x_lims.update(xlim_func())
    xs = np.linspace(min(x_lims), max(x_lims), 300)
    for i, (geometry, modes) in enumerate(geometries.items()):
        plot_modes(modes,
                   geometry,
                   coords=dict(row=i + 1, col=1),
                   label=(geometry == 'square'),
                   xs=xs)  # x axis changes))

        # for i, g in enumerate(folders):
        x = '' if not i else i + 1
        fig['layout'][f'yaxis{x}']['title'] = adjectives[geometry] + ' facet'

    st.plotly_chart(fig, use_column_width=True)
r'''
__Description of parameters__

__Circle__

f: facet fraction.

The ratio of facet diameter to spherical nanoparticle diameter.
        Range: 0.15-0.6
---------------------------------------------------------------------------
D (nm): Sphere's Diameter.

        Range: 40-100nm        
---------------------------------------------------------------------------
t (nm): gap thickness. 

        Range: 0.75-6nm
---------------------------------------------------------------------------
n: gap refractive index. 

        Range: 1.25-2
---------------------------------------------------------------------------


__Triangle__

_f_: facet fraction.

Analoguous to the ratio of facet diameter to spherical nanoparticle diameter.
$f = (fs/a)\frac{\sqrt[4]{3}}{2\sqrt{2(1+\sqrt{2})}}$,
where a is Rhombicuboctohedral side length, and fs is the facet side length.
This definition was chosen to preserve the ratio of areas on the facet to the middle 
cross-section of the nanoparticle in the spherical and rhombicuboctohedral cases.
for a regular rhombicuboctohedron, use $\frac{\sqrt[4]{3}}{2\sqrt{2(1+\sqrt{2})}} \simeq 0.3$
    Range: 0.15 - 0.6
---------------------------------------------------------------------------
_D_ (nm): roughly equivalent to Diameter.

A sphere of diameter D and Rhombicuboctohedron
defined by parameter D have the same cross-sectional area.
D = $a\sqrt{8(1+\sqrt{2})/\pi}\simeq 2.48a$
    
        Range: 40-100nm
---------------------------------------------------------------------------

__Square__

_f_: facet fraction.

Analoguous to the ratio of facet diameter to spherical nanoparticle diameter.
f = $(fs/a)(1/(2(1+\sqrt{2})))$,
where a is Rhombicuboctohedral side length, and fs is the facet side length.
This definition was chosen to preserve the ratio of areas on the facet to the middle 
cross-section of the nanoparticle in the spherical and rhombicuboctohedral cases.
for a regular rhombicuboctohedron, use $1/2(1+\sqrt{2}) \simeq 0.46$
        Range: 0.15 - 0.6
---------------------------------------------------------------------------
_D_ (nm): roughly equivalent to Diameter.

A sphere of diameter D and Rhombicuboctohedron
defined by parameter D have the same cross-sectional area.
D = $a\sqrt{8(1+\sqrt{2})/\pi} \simeq 2.48a$
    
        Range: 40-100nm

----------------------
All geometries have 5nm radius rounding applied to the bottom facet edge.


The units of the polynomial fits are as in the parameters above. 

real: y (nm) = polynomial(f, D, t, n)

imag: efficiency (unitless) = polynomial(D, real (eV), [f])

(f is an extra regressor for the 20 mode)
'''
