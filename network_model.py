#!/usr/bin/env python
# coding: utf-8
import pandapower.networks as pn
import pandapower.plotting as plot
import pandapower as pp
import numpy as np

def plot_network(net):
	#  función que recibe una red y la grafica.
    colors = ["k","b", "g", "r", "c", "y"]
    buses = net.bus.index.tolist() 
    coords = zip(net.bus_geodata.x.loc[buses].values+0.02, net.bus_geodata.y.loc[buses].values+0.02)
    lc = plot.create_line_collection(net, net.line.index, color=colors[0], use_bus_geodata=True, zorder=1)
    bc = plot.create_bus_collection(net, net.bus.index, size=0.02, bus_geodata=None ,color=colors[0], zorder=2)
    bic = plot.create_annotation_collection(size=0.07, texts=np.char.mod('%d', buses), coords=coords, zorder=3, color=colors[0])
    tf = plot.create_trafo_collection(net, trafos=net.trafo.index, color=colors[0], size=0.04, zorder=4)
    sc = plot.create_bus_collection(net, net.ext_grid.bus.values, patch_type="rect", size=0.03, color=colors[2], zorder=5)
    ld = plot.create_load_collection(net, size=0.03)
    sg = plot.create_sgen_collection(net, sgens=net.sgen.index[1:], size=0.05, orientation=0)
    sg1 = plot.create_sgen_collection(net, sgens=[net.sgen.index[0]], size=0.05, orientation=np.pi/2)
    sw = plot.create_line_switch_collection(net, size=0.04, distance_to_bus=0.08)
    return plot.draw_collections([lc, bc, bic, tf, sc, ld, sg, sg1, sw], figsize=(7,10))

def create_network():
	# carga la red de cigre de baja tensión y la modifica acrode a nuestros interese
	# retorna un objeto pandapower y la matriz de resistencias r y de reactancias x
	net = pn.create_cigre_network_lv()

	# Eliminar cargas que no se usarán
	for index in net.load.index:
		if not net.load['name'][index][5]=='R':
			net.load.drop(index, inplace=True)
	net.load.drop(0, inplace=True)

	# Eliminar todos los switches y crear uno nuevo
	for index in net.switch.index:
		net.switch.drop(index, inplace=True)
	pp.create_switch(net, bus=2, element=0, et='l', closed=True, type='CB', z_ohm=0.0)
	pp.create_switch(net, bus=5, element=3, et='l', closed=True, type='CB', z_ohm=0.0)
        
	# Eliminar trafos que no se usarán
	for index in net.trafo.index:
		if not net.trafo['name'][index]=='Trafo R0-R1':
			net.trafo.drop(index, inplace=True)
			
	# Eliminar lineas que no se usarán
	for index in net.line.index:
		if not net.line['name'][index][5]=='R':
			net.line.drop(index, inplace=True)
			
	# Eliminar buses que no se usarán
	for index in net.bus.index:
		if not net.bus['name'][index][4]=='R':
			net.bus.drop(index, inplace=True)
			net.bus_geodata.drop(index, inplace=True)
        
	# Cambio de lugar de la external grid
	net.ext_grid['bus']=1

	# Agregar nodo 20 y linea respectiva
	pp.create_bus(net, vn_kv=0.4, index=20, type='m', zone='CIGRE_LV', in_service=True, geodata=(7,-1))
	pp.create_line_from_parameters(net, from_bus=5, to_bus=20, std_type='UG3', length_km=0.030,
								   r_ohm_per_km=0.822, x_ohm_per_km=0.0847, c_nf_per_km=0.0, 
								   g_us_per_km=0.0, max_i_ka=1.0)

	# Agregar GD
	pp.create_sgen(net, bus=16, p_mw=0.030, q_mvar=0.0, name='Microturbina', scaling=1.0, type='CHP', index=1,
				  in_service=False)
	pp.create_sgen(net, bus=17, p_mw=0.010, q_mvar=0.0, name='WT', scaling=1.0, type='WP', index=2,
				  in_service=False)
	pp.create_sgen(net, bus=17, p_mw=0.010, q_mvar=0.0, name='PV1', scaling=1.0, type='PV', index=3,
				  in_service=True)
	pp.create_sgen(net, bus=18, p_mw=0.003, q_mvar=0.0, name='PV2', scaling=1.0, type='PV', index=4,
				  in_service=True)
	pp.create_sgen(net, bus=19, p_mw=0.010, q_mvar=0.0, name='Celda combustible', scaling=1.0, type='CHP', index=5,
				  in_service=False)
	pp.create_storage(net, bus=20, p_mw= 0.03, max_e_mwh=0.060, q_mvar=0, min_e_mwh=0.012, in_service=False)

	# Cambiar consumos
	S_loads = [0.015, 0.072, 0.050, 0.015, 0.047]
	cos_phi = 0.85
	for i in net.load.index:
		net.load['p_mw'][i] = S_loads[i-1]*cos_phi
		net.load['q_mvar'][i] = S_loads[i-1]*np.sqrt(1-np.square(cos_phi))

	# Modificar Geodata de los buses para graficar
	y = np.array([11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 8, 7, 7, 7, 6, 5, 2, 1, 7])
	x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 2, 3, 3, -1, 1, -1, -1])
	net.bus_geodata['x'] = 0.3*x
	net.bus_geodata['y'] = 0.3*y
	
	Zn = ((0.4*1e3)**2)/(net.sn_mva*1e6)
	r = np.zeros((len(net.bus), len(net.bus)))
	x = np.zeros((len(net.bus), len(net.bus)))
	
	for index in net.line.index:
		r_line = net.line['r_ohm_per_km'][index]*net.line['length_km'][index]/Zn
		x_line = net.line['x_ohm_per_km'][index]*net.line['length_km'][index]/Zn
		r[net.line['from_bus'][index]-1,net.line['to_bus'][index]-1] = r_line
		x[net.line['from_bus'][index]-1,net.line['to_bus'][index]-1] = x_line
		r[net.line['to_bus'][index]-1,net.line['from_bus'][index]-1] = r_line
		x[net.line['to_bus'][index]-1,net.line['from_bus'][index]-1] = x_line

	return net, r, x

def calculate_current(v1,t1,v2,t2, r, x):
   # calcula la corriente entre  nodos dadas las tensiones y angulos en 
   # con impedancia r+jx entre ellos
   V1 = v1*np.cos(np.deg2rad(t1)) + v1*np.sin(np.deg2rad(t1))*1j
   V2 = v2*np.cos(np.deg2rad(t2)) + v2*np.sin(np.deg2rad(t2))*1j
   I = (V1-V2)/(r+x*1j)
   return I
       
def observation_model(X, r, x):
   # calcua las potencias que deberían observarse con los valores de tensiones y 
   # ángulos del estado X usando los parametros r y x de las lineas
   
   # Potencia nodo 2
   I_23 = calculate_current(X[1],X[21],X[2],X[22],r[1,2],x[1,2])
   S_2 = (X[1]*np.cos(np.deg2rad(X[21])) + X[1]*np.sin(np.deg2rad(X[21]))*1j)*np.conj(I_23)
   P2, Q2 = np.real(S_2), np.imag(S_2)
   # Potencia nodo 12
   I_34 = calculate_current(X[2],X[22],X[3],X[23],r[2,3],x[2,3])
   I_45 = calculate_current(X[3],X[23],X[4],X[24],r[3,4],x[3,4])
   I_124 = I_45 - I_34
   S_12 = (X[11]*np.cos(np.deg2rad(X[31])) + X[11]*np.sin(np.deg2rad(X[31]))*1j)*np.conj(I_124)
   P12, Q12 = np.real(S_12), np.imag(S_12)
   # Potencia nodo 16
   I_1514 = calculate_current(X[14],X[34],X[13],X[33],r[14,13],x[14,13])
   S_16 = (X[15]*np.cos(np.deg2rad(X[35])) + X[15]*np.sin(np.deg2rad(X[35]))*1j)*np.conj(I_1514)
   P16, Q16 = np.real(S_16), np.imag(S_16)
   # Potencia nodo 17
   I_67 = calculate_current(X[5],X[25],X[6],X[26],r[5,6],x[5,6])
   I_78 = calculate_current(X[6],X[26],X[7],X[27],r[6,7],x[6,7])
   I_177 = I_78 - I_67
   S_17 = (X[16]*np.cos(np.deg2rad(X[36])) + X[16]*np.sin(np.deg2rad(X[36]))*1j)*np.conj(I_177)
   P17, Q17 = np.real(S_17), np.imag(S_17)
   # Potencia nodo 18
   I_910 = calculate_current(X[8],X[28],X[9],X[29],r[8,9],x[8,9])
   I_1011 = calculate_current(X[9],X[29],X[10],X[30],r[9,10],x[9,10])
   I_1810 = I_910 - I_1011
   S_18 = (X[17]*np.cos(np.deg2rad(X[37])) + X[17]*np.sin(np.deg2rad(X[37]))*1j)*np.conj(I_1810)
   P18, Q18 = np.real(S_18), np.imag(S_18)
   # Potencia nodo 19
   I_1911 = calculate_current(X[18],X[38],X[10],X[30],r[18,10],x[18,10])
   S_19 = (X[18]*np.cos(np.deg2rad(X[38])) + X[18]*np.sin(np.deg2rad(X[38]))*1j)*np.conj(I_1911)
   P19, Q19 = np.real(S_19), np.imag(S_19)
   # Potencia nodo 20
       # I_45 calculado antes
   I_513 = calculate_current(X[4],X[24],X[12],X[32],r[4,12],x[4,12])
   I_56 = calculate_current(X[4],X[24],X[5],X[25],r[4,5],x[4,5])
   I_205 = I_513 + I_56 - I_45
   S_20 = (X[19]*np.cos(np.deg2rad(X[39])) + X[19]*np.sin(np.deg2rad(X[39]))*1j)*np.conj(I_205)
   P20, Q20 = np.real(S_20), np.imag(S_20)
   
   return np.array([P2, -P12, -P16, -P17, P18, -P19, P20, Q2, -Q12, -Q16, -Q17, Q18, -Q19, -Q20])

