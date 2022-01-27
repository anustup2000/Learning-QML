#!/usr/bin/env python
# coding: utf-8

# In[1]:


""" 
MIT License
Copyright (c) 2022 Maxime Dion <maxime.dion@usherbrooke.ca>
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import plotly.graph_objects as go
import numpy as np

from qiskit import Aer, execute

class_A_color = '#636EFA'
class_B_color = '#EF553B'
good_color = '#00CC96'
bad_color = '#EF553B'
black_color = '#000000'
bloch_sphere_color = '#C0C0C0'

def data_figure(data_xs,data_ys):

    fig = go.FigureWidget()

    trace_a = go.Scatter(
        x = data_xs[data_ys == 0,0],
        y = data_xs[data_ys == 0,1],
        name = 'A', 
        mode='markers',
        marker=dict(color=class_A_color, size=10, showscale=False)
        )
    
    trace_b = go.Scatter(
        x = data_xs[data_ys == 1,0],
        y = data_xs[data_ys == 1,1],
        name = 'B', 
        mode='markers',
        marker=dict(color=class_B_color, size=10, showscale=False)
        )

    fig.add_trace(trace_a)
    fig.add_trace(trace_b)

    fig.update_layout(
        width=500, 
        height=500,
        xaxis_title="x1",
        yaxis_title="x2",
        legend_title="Classes",
        )

    return fig

def classification_figure(data_xs,data_ys,predictions_ys):

    good = data_ys == predictions_ys
    accuracy = np.sum(good)/len(good)

    fig = go.FigureWidget()

    trace_good = go.Scatter(
        x = data_xs[good,0],
        y = data_xs[good,1],
        name = 'Good', 
        mode='markers',
        marker=dict(color=good_color, size=10, showscale=False)
        )
    
    trace_bad = go.Scatter(
        x = data_xs[~good,0],
        y = data_xs[~good,1],
        name = 'Bad', 
        mode='markers',
        marker=dict(color=bad_color, size=10, showscale=False)
        )

    fig.add_trace(trace_good)
    fig.add_trace(trace_bad)

    fig.update_layout(
        width=500, 
        height=500,
        xaxis_title="x1",
        yaxis_title="x2",
        title=f'Classification ({np.sum(good)} good, {np.sum(good == False)} bad, A = {accuracy*100:.2f}%)'
        )

    return fig

def history_figure(history):

    xys = np.zeros((len(history),2))
    
    for i, (nb_fct_eval,params, value) in enumerate(history):
        xys[i,0] = nb_fct_eval
        xys[i,1] = value
    
    fig = go.Figure(data=go.Scatter(x=xys[:,0], y=xys[:,1]))

    return fig

def bloch_sphere_statevector_figure(statevectors = None,data_ys = None):

    r = 0.98
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    sphere_x = r * np.sin(v) * np.cos(u)
    sphere_y = r * np.sin(v) * np.sin(u)
    sphere_z = r * np.cos(v)

    fig = go.FigureWidget()

    surfacecolor = np.zeros(shape=sphere_x.shape) 
    colorscale = [[0, bloch_sphere_color], 
              [1, bloch_sphere_color]]

    bloch_surface = go.Surface(
        x=sphere_x, y=sphere_y, z=sphere_z, surfacecolor = surfacecolor,
        colorscale=colorscale,
        name = 'Sphere', 
        showscale=False,
        opacity=0.5,
        lighting=dict(ambient=1),
        )

    axis_dots_xyzs = np.array([[r,0,0],[0,r,0],[0,0,r],[-r,0,0],[0,-r,0],[0,0,-r]])

    theta = 2*np.pi*np.linspace(0,1)
    cos_t = r*np.cos(theta)
    sin_t = r*np.sin(theta)
    zero_t = np.zeros(theta.shape)
    traces_mesh = list()
    traces_mesh.append( 
        go.Scatter3d(x=cos_t, y=sin_t, z=zero_t,
        name = 'xy plane',mode='lines',showlegend=False,line=dict(color=black_color,width=2))
        )
    traces_mesh.append( 
        go.Scatter3d(y=cos_t, z=sin_t, x=zero_t,
            name = 'yz plane',mode='lines',showlegend=False,line=dict(color=black_color,width=2))
        )
    traces_mesh.append( 
        go.Scatter3d(z=cos_t, x=sin_t, y=zero_t,
            name = 'zx plane',mode='lines',showlegend=False,line=dict(color=black_color,width=2))
        )
    traces_mesh.append( 
        go.Scatter3d(x=[0,0], y=[0,0], z=[-r,r],
            name = 'z axis',mode='lines',showlegend=False,line=dict(color=black_color,width=2))
        )
    traces_mesh.append( 
        go.Scatter3d(y=[0,0], z=[0,0], x=[-r,r],
            name = 'x axis',mode='lines',showlegend=False,line=dict(color=black_color,width=2))
        )
    traces_mesh.append( 
        go.Scatter3d(z=[0,0], x=[0,0], y=[-r,r],
            name = 'y axis',mode='lines',showlegend=False,line=dict(color=black_color,width=2))
        )
    traces_mesh.append( 
        go.Scatter3d(x=np.sqrt(0.5)*sin_t, y=np.sqrt(0.5)*sin_t, z=cos_t,
            name = 'x-y plane',mode='lines',showlegend=False,line=dict(color=black_color,width=1))
        )
    traces_mesh.append( 
        go.Scatter3d(x=np.sqrt(0.5)*sin_t, y=-np.sqrt(0.5)*sin_t, z=cos_t,
            name = 'x+y plane',mode='lines',showlegend=False,line=dict(color=black_color,width=1))
        )

    if statevectors is not None:

        xyzs = statevectors_to_xyzs(statevectors)

        mask_a = data_ys == 0
        trace_a = go.Scatter3d(
            x=xyzs[mask_a,0], y=xyzs[mask_a,1], z=xyzs[mask_a,2],
            name = 'A', 
            mode='markers',
            marker=dict(color=class_A_color, size=8, showscale=False)
            )

        mask_b = data_ys == 1
        trace_b = go.Scatter3d(
            x=xyzs[mask_b,0], y=xyzs[mask_b,1], z=xyzs[mask_b,2],
            name = 'B', 
            mode='markers',
            marker=dict(color=class_B_color, size=8, showscale=False)
            )

    trace_axis_dots = go.Scatter3d(
        x=axis_dots_xyzs[:,0], y=axis_dots_xyzs[:,1], z=axis_dots_xyzs[:,2], 
        mode='markers+text',showlegend=False,
        marker=dict(color=black_color, size=4, showscale=False)
        )
    
    scale_outside = 1.15
    trace_axis_labels = go.Scatter3d(
        x=scale_outside*axis_dots_xyzs[:,0], y=scale_outside*axis_dots_xyzs[:,1], z=scale_outside*axis_dots_xyzs[:,2], 
        mode='text',showlegend=False,
        text = ['X','Y','|0>','-X','-Y','|1>'],
        textposition = "middle center",
        textfont = dict(size=20),
        marker=dict(color=black_color, size=4, showscale=False)
        )

    fig.add_trace(bloch_surface)
    for trace in traces_mesh:
        fig.add_trace(trace)

    fig.add_trace(trace_axis_dots)
    if statevectors is not None:
        fig.add_trace(trace_a)
        fig.add_trace(trace_b)

    fig.add_trace(trace_axis_labels)

    max_lim = 1.2
    fig.update_layout(
        legend=dict(
            x=0,y=.5,
            traceorder="normal",
            font=dict(family="sans-serif",size=20,color="black"),
        ),
        height=500,width=500,
        margin=dict(l=0, r=0, b=0, t=0),
        )

    fig.update_scenes(
        aspectratio=dict(x=1, y=1, z=1),
        xaxis = dict(nticks=4, range=[-max_lim,max_lim],),
        yaxis = dict(nticks=4, range=[-max_lim,max_lim],),
        zaxis = dict(nticks=4, range=[-max_lim,max_lim],),
        xaxis_visible=False, yaxis_visible=False,zaxis_visible=False
        )

    return fig


def circuits_to_statevectors(circuits):

    statevector_simulator = Aer.get_backend('statevector_simulator')

    n_circuits = len(circuits)
    result = execute(circuits, statevector_simulator).result()
    statevectors = np.zeros((n_circuits,2),dtype = complex)
    for i in range(n_circuits):
        statevectors[i,:] = result.get_statevector(i)

    return statevectors


def statevectors_to_xyzs(statevectors):
    
    phis = np.angle(statevectors[:,1]) - np.angle(statevectors[:,0])
    thetas = np.arccos(np.abs(statevectors[:,0])) + np.arcsin(np.abs(statevectors[:,1]))
    xyzs = np.zeros((statevectors.shape[0],3))
    xyzs[:,0] = np.sin(thetas) * np.cos(phis)
    xyzs[:,1] = np.sin(thetas) * np.sin(phis)
    xyzs[:,2] = np.cos(thetas)

    return xyzs


# In[ ]:




