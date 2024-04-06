# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:15:27 2023

节点的生成逻辑还存在问题，当前的逻辑为识别轮廓得到的点和判断交叉得到的点取交集，实际上可能存在一条边上的一系列点
他们只能通过识别轮廓得到，而不能通过判断交叉得到。

@author: A
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import defaultdict

######### FIXED PARAMETERS ######### 
PI = np.pi
img_width = 1280
img_height = 640
# colors = np.loadtxt('../02_data/color.csv',dtype=np.int32,delimiter=',')
# colors = colors[:,[2,1,0]]


########## VARIED PARAMETERS #######
# name of image
# k_img = '18' 
# parameters for connecting lines
beam_width = 8
colu_width = 15
if_detect_diag_line = 1
degGap = 6
# parameters for finding intersection nodes
max_dist = 24
# parameters for deleting clustered nodes 
clstRad = 15



def detect_walls(img,if_show = False):
    img = img.copy()
    white = 255*np.ones((img_height,img_width,3),dtype=np.uint8)
    
    # img = cv2.pyrDown(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(13,13),0)    
    ret, thresh = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY_INV)
    white = 255-thresh
    contours, hier = cv2.findContours(thresh,cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        # find bounding box coordinates
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(white, [box], 0, (0,0, 255), 2)
        # x,y,w,h = cv2.boundingRect(c)
        # rects.append([x,y,w,h])
        # cv2.rectangle(white,(x,y),(x+w,y+h),(0,255,0),1)
    
    if if_show:
        cv2.imwrite('columns.png',white)
        cv2.imshow('thresh',white)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print('walls')
    return thresh

def detect_columns(img):
    img = img.copy()
    # img = cv2.pyrDown(img)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(13,13),0)
    
    ret, thresh = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(thresh,cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        # find bounding box coordinates
        x,y,w,h = cv2.boundingRect(c)
        rects.append([x,y,w,h])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

    # cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
    rects = np.array(rects)
    return rects

def detect_columns_wall(thresh):
    thresh = thresh.copy()
    contours, hier = cv2.findContours(thresh,cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for c in contours:
        # find bounding box coordinates
        x,y,w,h = cv2.boundingRect(c)
        rects.append([x,y,w,h])
    rects = np.array(rects)
    return rects

def delete_walls(img, if_show=False):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    inverse = 255-gray
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(inverse,kernel,iterations = 1)
    
    if if_show:
        cv2.imshow('thresh',gray+erosion)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    
    return gray+erosion

def draw_ori_lines(img,lines):
    imgCopy = img.copy()
    white = 255*np.ones((img_height,img_width,3),dtype=np.uint8)
    id_clus=0
    for line in lines:
        for x1, y1, x2, y2 in line:
            x1 = np.int32(x1)
            x2 = np.int32(x2)
            y1 = np.int32(y1)
            y2 = np.int32(y2)
            # color_cur = (int(colors[id_clus % 10,0]),int(colors[id_clus % 10,1]),int(colors[id_clus % 10,2]))
            grey_cur = random.randint(0,128)
            color_cur = (grey_cur,grey_cur,grey_cur)
            cv2.line(white,(x1,y1),(x2,y2),color_cur,1,cv2.LINE_AA)
            id_clus += 1
    cv2.imwrite('84-lines-ori.png',white)
    cv2.imshow('lines_ori',white)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    return 

def draw_cnt_lines(img,HLS,VLS,DLS):

    imgCopy = img.copy()
    white = 255*np.ones((img_height,img_width,3),dtype=np.uint8)
    id_clus = 0
    for line in HLS:
        x1,y1,x2,y2 = tuple(line)
        grey_cur = random.randint(0,128)
        color_cur = (grey_cur,grey_cur,grey_cur)
        # color_cur = (int(colors[id_clus % 10,0]),int(colors[id_clus % 10,1]),int(colors[id_clus % 10,2]))
        cv2.line(white,(x1,y1),(x2,y2),color_cur,2,cv2.LINE_AA)
        id_clus += 1
    for line in VLS:
        x1,y1,x2,y2 = tuple(line)
        grey_cur = random.randint(0,128)
        color_cur = (grey_cur,grey_cur,grey_cur)
        # color_cur = (int(colors[id_clus % 10,0]),int(colors[id_clus % 10,1]),int(colors[id_clus % 10,2]))
        cv2.line(white,(x1,y1),(x2,y2),color_cur,2,cv2.LINE_AA)
        id_clus += 1
    for deg in DLS:
        lines = DLS[deg]
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2 = tuple(line)
                grey_cur = random.randint(0,128)
                color_cur = (grey_cur,grey_cur,grey_cur)
                # color_cur = (int(colors[id_clus % 10,0]),int(colors[id_clus % 10,1]),int(colors[id_clus % 10,2]))
                cv2.line(white,(x1,y1),(x2,y2),color_cur,2,cv2.LINE_AA)
                id_clus += 1
    cv2.imwrite('84-lines.png',white)
    cv2.imshow('lines',white)
    cv2.waitKey()
    cv2.destroyAllWindows()

def preprocess_lines(lines):

    # Divide the lines into Horizontal lines, Vertical lines and Diagonal lines
    Hlines = []
    Vlines = []
    Dlines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1-x2)<=3:
                Vlines.append([x1,y1,x2,y2])
            elif abs(y1-y2)<=3:
                Hlines.append([x1,y1,x2,y2])
            else:
                Dlines.append([x1,y1,x2,y2])
    Hlines = np.array(Hlines)
    Vlines = np.array(Vlines)
    Dlines = np.array(Dlines)
    
    # sort the horizontal lines by the y coordinate, top-down
    Hlines = Hlines[Hlines[:,1].argsort()]
    # sort the vertical lines by the x coordinate, leftmost-right
    Vlines = Vlines[Vlines[:,0].argsort()]
    
    dlinesDeg = {}
    dlinesDict ={}
    
    # divide all the diagonal lines into sets 按照是否平行.
    for line in Dlines:
        x1,y1,x2,y2 = line
        deg = np.arctan((y2-y1)/(x2-x1))/PI*180
        dlinesDeg[deg]=line
    '''后续应该修改将线分类的方法， 以不影响水平线和竖直线的识别为最高原则
    即，当线离水平线很接近时，应该直接设置为水平线'''
    degs = sorted(dlinesDeg)
    dlines = []
    degTot = 0
    degAvg = 0
    for deg in sorted(dlinesDeg):
        if len(dlines) <1 or deg-degAvg < degGap:
            dlines.append(dlinesDeg[deg])
            degTot += deg
        else:
            if len(dlines)>0:
                dlinesDict[degAvg] = dlines
            dlines =[dlinesDeg[deg]]
            degTot = deg
        degAvg = np.int32(degTot/len(dlines))
    if len(dlines)>0:
        dlinesDict[degAvg] = dlines
        
    return Hlines, Vlines, dlinesDict

def rotate(deg):
    Cos = np.cos(deg/180*PI)
    Sin = np.sin(deg/180*PI)
    Trans = np.array([[Cos,-Sin,  0,   0],
                      [Sin, Cos,  0,   0],
                      [  0,   0,Cos,-Sin],
                      [  0,   0,Sin, Cos]])
    
    return Trans

def connect_lines(Lines,beam_width = 10, max_dist = 15, deg =0):
    '''

    Parameters
    ----------
    Lines : ndarray,[[x1,y1,x2,y2],...,[x1,y1,x2,y2]]
        end points of lines
    beam_width : int, optional
        key parameter, any lines within this distance are put into the same cluster.
        This parameter represents a distance little wider than beam_width
        The default is 10.
    max_dist : int, optional
        DESCRIPTION. The default is 15.
        key parameter, if the gap between two line segments on a same line
        is smaller than this distance, they are connected together
    deg : float, optional
        The direction of th lines.
        For Horizontal lines deg =0; For Vertical lines deg = -90
        The default is 0.

    Returns
    -------
    None.

    '''

    Trans = rotate(deg)
    Lines = Lines @ Trans
    Lines = Lines[Lines[:,1].argsort()]
    # divide the horizontal lines into clusters according to the y coordinate
    CLST = {}
    id_clus = 0
    
    CLST[id_clus]=[Lines[0]] # initiate

    for i in range(1,len(Lines)):
        #########这句可能有问题############
        if Lines[i,1]-Lines[i-1,1]<=beam_width:
            CLST[id_clus].append(Lines[i])
        else:
            CLST[id_clus] = np.array(CLST[id_clus])
            id_clus += 1
            CLST[id_clus]=[Lines[i]]

    CLST[id_clus]=np.array(CLST[id_clus])
    
    # connect the lines in each clusters together
    CNCT = {}
    id_clus = 0
    CNCT[id_clus]=[]
    
    for id_clus in range(len(CLST)):
        lines_cur = CLST[id_clus]
    
        for i in range(len(lines_cur)):
            
            # swap the order of two end points of a segment if they are inverted
            if lines_cur[i,0]>lines_cur[i,2]:
                tmp = lines_cur[i,0]
                lines_cur[i,0] = lines_cur[i,2]
                lines_cur[i,2] = tmp
        # sort the lines in the current cluster by x coordinate
        lines_cur = lines_cur[lines_cur[:,0].argsort()]
        # the y coordinate is set to be the average value of all segments in this cluster
        y_avg = lines_cur[:,[1,3]].mean()
    
        x1,y1,x2,y2 = tuple(lines_cur[0])
        x1_cur = x1
        x2_cur = x2
        CNCT[id_clus]=[]
        for line in lines_cur[1:,:]:
            x1,y1,x2,y2 = tuple(line)
            # the gap between the right end of the left segment and the left end of the
            # right segment is smaller than max_dist
            if x1 < x2_cur + max_dist:
                x2_cur = x2
            else:
                if x2_cur-x1_cur >max_dist:
                    CNCT[id_clus].append([x1_cur,y_avg,x2_cur,y_avg])
                    
                x1_cur = x1
                x2_cur = x2
        if x2_cur-x1_cur >max_dist:
            CNCT[id_clus].append([x1_cur,y_avg,x2_cur,y_avg])
        if len(CNCT[id_clus])>0:
            CNCT[id_clus] = np.array(CNCT[id_clus],dtype=np.int32)
        else:
            CNCT.pop(id_clus)
    
    # rotate the lines back
    Trans =  rotate(-deg)
    HLS = np.zeros((4,),dtype=np.int32)
    for id_clus in CNCT:
        HLS = np.vstack((HLS,CNCT[id_clus]))
    HLS = HLS @ Trans
    HLS = HLS.astype('int32')
    if len(HLS.shape)>1:
        return HLS[1:]
    else:
        return None

def connect_all_lines(Hlines,Vlines,dlinesDict):
    HLS = connect_lines(Hlines,beam_width=beam_width,max_dist=colu_width)
    VLS = connect_lines(Vlines,beam_width=beam_width,max_dist=colu_width,deg=90)
    DLS = {}
    for deg in dlinesDict:
        DLS[deg] = connect_lines(dlinesDict[deg],beam_width=beam_width,max_dist=colu_width,deg=deg)
        if DLS[deg] is None:
            DLS.pop(deg)
    
    if not if_detect_diag_line: DLS={}
    
    return HLS, VLS, DLS

def Intersection(line1:tuple,line2:tuple,maxDist=35)->tuple:
    # 首先在line1的两端延长，并判断延长后的点是否在line2两侧
    x1,y1,x2,y2 = line1
    len1 = np.sqrt((x2-x1)**2+(y2-y1)**2)
    x2t = maxDist/len1*(x2-x1)+x2
    y2t = maxDist/len1*(y2-y1)+y2
    x1t = maxDist/len1*(x1-x2)+x1
    y1t = maxDist/len1*(y1-y2)+y1
    
    x3,y3,x4,y4 = line2
    len2 = np.sqrt((x4-x3)**2+(y4-y3)**2)
    x4t = maxDist/len2*(x4-x3)+x4
    y4t = maxDist/len2*(y4-y3)+y4
    x3t = maxDist/len2*(x3-x4)+x3
    y3t = maxDist/len2*(y3-y4)+y3
    
    dx1 = x2-x1
    dy1 = y2-y1
    dx2 = x4-x3
    dy2 = y4-y3
    
    
    
    
    if ((x1t-x3)*(y4-y3)-(x4-x3)*(y1t-y3))*((x2t-x3)*(y4-y3)-(x4-x3)*(y2t-y3))<0 \
        and ((x3t-x1)*(y2-y1)-(x2-x1)*(y3t-y1))*((x4t-x1)*(y2-y1)-(x2-x1)*(y4t-y1))<0:
        A = np.array([[dy1,-dx1],[dy2,-dx2]])
        B = np.array([x1*dy1-y1*dx1,x3*dy2-y3*dx2])
        X = np.linalg.solve(A,B)
        return np.array(X,dtype=np.int32)
    else:
        return None

def find_intersection_node(HLS,VLS,DLS):
    Nodes = []
    edgesDict = {}
    # 为每条线初始化一个字典，key是tuple
    # edgesDict[(x1,y1,x2,y2)] ={(node1x,node1y):nodeID1,...,(nodenx,nodeny):nodeIDn}
    for hline in HLS:
        edgesDict[tuple(hline)]={}
    for vline in VLS:
        edgesDict[tuple(vline)]={}
    for deg in DLS:
        for dline in DLS[deg]:
            edgesDict[tuple(dline)]={}
    
        
    nodeID = 0
    nodesDict = defaultdict(list)
    # 同样由max_dist 判断是否相交
    for hline in HLS:
        for vline in VLS:
            if vline[0]>hline[0]-max_dist and vline[0]<hline[2]+max_dist \
                and hline[1]>vline[1]-max_dist and hline[1]<vline[3]+max_dist:
                node = np.array([vline[0],hline[1]],dtype=np.int32)
                Nodes.append(node)
                edgesDict[tuple(hline)][tuple(node)] = nodeID
                edgesDict[tuple(vline)][tuple(node)] = nodeID
                nodesDict[tuple(node)].append(hline)
                nodesDict[tuple(node)].append(vline)
                nodeID += 1
        for deg in DLS:
            for dline in DLS[deg]:
                node = Intersection(hline,dline,max_dist)
                if node is not None:
                    Nodes.append(node)
                    edgesDict[tuple(hline)][tuple(node)] = nodeID
                    edgesDict[tuple(dline)][tuple(node)] = nodeID
                    nodesDict[tuple(node)].append(hline)
                    nodesDict[tuple(node)].append(dline)
                    nodeID += 1
    for vline in VLS:
        for deg in DLS:
            for dline in DLS[deg]:
                node = Intersection(vline,dline,max_dist)
                if node is not None:
                    Nodes.append(node)
                    edgesDict[tuple(vline)][tuple(node)] = nodeID
                    edgesDict[tuple(dline)][tuple(node)] = nodeID
                    nodesDict[tuple(node)].append(vline)
                    nodesDict[tuple(node)].append(dline)
                    nodeID += 1
    degs = sorted(DLS)
    for i in range(len(degs)-1):
        for dline1 in DLS[degs[i]]:
            for j in range(i+1,len(degs)):
                for dline2 in DLS[degs[j]]:
                    node = Intersection(dline1,dline2,max_dist)
                    if node is not None:
                        Nodes.append(node)
                        edgesDict[tuple(dline1)][tuple(node)] = nodeID
                        edgesDict[tuple(dline2)][tuple(node)] = nodeID
                        nodesDict[tuple(node)].append(dline1)
                        nodesDict[tuple(node)].append(dline2)
                        nodeID += 1
            
    
    Nodes = np.array(Nodes,dtype = np.int32)
    
    return Nodes, nodesDict, edgesDict

def delete_dupli_nodes(Nodes,nodesDict,edgesDict):
    

    nodeCLST = defaultdict(list)
    
    for node in Nodes:
        minDist = img_width
        centID = None
        for centroid in nodeCLST:
            nodeDist = np.sqrt((node[0]-centroid[0])**2+(node[1]-centroid[1])**2)
            if nodeDist < minDist: #更新
                minDist = nodeDist
                centID = centroid
        # 如果当前节点和某一个节点聚类的质心距离小于阈值，就将其归入该聚类中，并更新该聚类的质心
        if minDist < clstRad:
            # 计算新的质心
            n = len(nodeCLST[centID])
            xAvg = np.int32((centID[0]*n + node[0])/(n+1))
            yAvg = np.int32((centID[1]*n + node[1])/(n+1))
            nodeCLST[(xAvg,yAvg)] = nodeCLST.pop(centID)
            nodeCLST[(xAvg,yAvg)].append(node)
        else:
        # 否则就将当前节点作为一个新的聚类放入nodeCLST中
            nodeCLST[tuple(node)].append(node)
    
    
    
    # 在每个聚类中，取平均值作为节点值，并修改相应的nodesDict 和 edgesDict
    nodesDense = []  # 去重后的节点列表
    nodeID = 0
    for key in nodeCLST:
        nodesDense.append(np.array(key))
        nodesCur = nodeCLST[key]
        if len(nodesCur)==1: #只有一个节点
            # print(f'key={key}')
            node = nodesCur[0]
            lines = nodesDict[tuple(node)]
            for line in lines:
                edgesDict[tuple(line)][tuple(node)] = nodeID
                # print(f'line={line},node={node},nodeID={nodeID}')
        
        else:
            for node in nodesCur:
                lines = nodesDict[tuple(node)]
                for line in lines:
                    if tuple(node) in edgesDict[tuple(line)]:
                        del edgesDict[tuple(line)][tuple(node)]
                    edgesDict[tuple(line)][key] = nodeID
        nodeID +=1
    
    return nodesDense

def draw_nodes(nodesDense,thresh,if_show = False):
    # draw nodes
    img_nodes = 255*np.ones((img_height,img_width),dtype=np.uint8)
    # radius = 10
    edge_len = 5
    for x,y in nodesDense:
        # center = (int(x),int(y))
        # draw the circle
        # cv2.circle(img_nodes,center,radius,(0,0,255),2)
        cv2.rectangle(img_nodes,(x-edge_len,y-edge_len),(x+edge_len,y+edge_len),(0,0,0),-1)
    
    raw_nodes = cv2.bitwise_and(thresh, 255-img_nodes)
    if if_show:
        cv2.imshow('nodes',img_nodes)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    return img_nodes,raw_nodes

def establish_graph(nodesDense,HLS,VLS,DLS,raw_nodes):
    Gstrc = nx.Graph()
    # establish nodes
    for i,node in enumerate(nodesDense):
        Gstrc.add_node(i,feature=node,label=0)
    
    # establish edges
    for line in HLS:
        lineDict = edgesDict[tuple(line)]
        sortedXY = sorted(lineDict)
        # 至少得有2个顶点
        if len(lineDict) <=1:
            continue
        for j in range(len(lineDict)-1):
            Gstrc.add_edge(lineDict[sortedXY[j]],lineDict[sortedXY[j+1]])
    
    for line in VLS:
        lineDict = edgesDict[tuple(line)]
        sortedXY = sorted(lineDict,key=lambda x: x[1])
        # 至少得有2个顶点
        if len(lineDict) <=1:
            continue
        for j in range(len(lineDict)-1):
            Gstrc.add_edge(lineDict[sortedXY[j]],lineDict[sortedXY[j+1]])
    
    for deg in DLS:
        for line in DLS[deg]:
            lineDict = edgesDict[tuple(line)]
            sortedXY = sorted(lineDict)
            # 至少得有2个顶点
            if len(lineDict) <=1:
                continue
            for j in range(len(lineDict)-1):
                Gstrc.add_edge(lineDict[sortedXY[j]],lineDict[sortedXY[j+1]])
    
    rects = detect_columns_wall(raw_nodes)
    cents = rects[:,0:2]+1/2*rects[:,2:4]
    
    for center in cents:
        close_ind = 0
        close_dist = 1e8
        for node in Gstrc.nodes(data=True):
            idn, attr = node
            cord = attr['feature']
            dist = np.linalg.norm(center-cord,ord=2)
            if dist < close_dist:
                close_dist = dist
                close_ind = idn
        Gstrc.nodes[close_ind]['label']=1
        
    return Gstrc

def exportGraph(G,filename):
    Adj = nx.to_numpy_array(G,dtype=np.int32)
    node_coord = []
    node_column= []
    for node in G.nodes(data=True):
        idn, attr = node
        center = attr['feature']
        node_coord.append(center)
        is_colm = attr['label']
        node_column.append(is_colm)
    node_coord = np.array(node_coord,dtype=np.int32)

    node_column = np.array(node_column,dtype=np.int32).reshape(-1,1)
    nodeX = np.hstack((node_coord,node_column))
    np.savetxt(filename+'-Adj.csv',Adj,delimiter=',')
    np.savetxt(filename+'-nodeX.csv',nodeX,delimiter=',')
    
    return Adj,nodeX

def draw_graph(graph,color='black',file_name=None):

    # draw graphs
    # nx.draw_networkx(graph)
    
    # white = 0*np.ones((img_height,img_width,3),dtype=np.uint8)
    
    # for node in graph.nodes(data=True):
    #     idn, attr = node
    #     center = tuple(attr['feature'])
    #     is_colm = attr['label']
    #     radius = 6
    #     if is_colm:
    #         # cv2.circle(white,center,radius,(0,0,0),-1)
    #         cv2.circle(white,center,radius,(255,255,255),-1)
    #     else:
    #         # cv2.circle(white,center,radius,(0,0,0),1)
    #         cv2.circle(white,center,radius,(255,255,255),-1)
    # i = 0
    # for edge in graph.edges():
    #     node1 = edge[0]
    #     node2 = edge[1]
    #     cord1 = tuple(graph.nodes[node1]['feature'])
    #     cord2 = tuple(graph.nodes[node2]['feature'])
        
    #     color_cur = (255,255,255)
        
    #     # color_cur = (int(colors[i % 10,0]),int(colors[i % 10,1]),int(colors[i % 10,2]))
        
    #     cv2.line(white,cord1,cord2,color_cur,2,cv2.LINE_AA)
    #     i = i+1
    
    # cv2.imshow('white',white)
    
    # cv2.imwrite(f'../02_data/white_sturc_{k_img}_mono.png',white)
    
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # 创建一个新的绘图
    fig, ax = plt.subplots()
    
    
    # 绘制边
    for edge in graph.edges():
        
        node1 = edge[0]
        node2 = edge[1]
        start_x, start_y = tuple(graph.nodes[node1]['feature'])
        end_x, end_y = tuple(graph.nodes[node2]['feature'])
        

        ax.plot([start_x, end_x], [-start_y, -end_y], color=color,linewidth=1,zorder=1)
    # 绘制节点
    for node in graph.nodes(data=True):
        idn, attr = node
        x,y = tuple(attr['feature'])
        is_colm = attr['label']
        radius = 6
        if is_colm == 1:
            ax.add_artist(plt.Circle((x, -y), radius, color=color, fill=True))
        else:
            
            ax.add_artist(plt.Circle((x, -y), radius, color=color,fill=False,linewidth=1,zorder=2))
            ax.add_artist(plt.Circle((x, -y), radius-2, color='#FBFBFB', fill=True,zorder=3))
    
    # 设置坐标轴范围
    plt.axis('equal')
    plt.axis('off')
    ax.set_aspect('equal')
    # 显示图形
    plt.show()
    fig.savefig(file_name+'.svg')#,dpi=300,transparent=True)

    
if __name__ == '__main__':
    
    data_dir = '../02_data/' # 输入的structual layout images
    sheet_dir = '../03_sheets_3/' # 输出的adj 和 nodex的文件夹
    graph_dir = '../04_output/' # 输出的graph vector plot 的文件夹
    
    if not os.path.exists(sheet_dir):
        os.makedirs(sheet_dir)
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    
    image_files = os.listdir(data_dir)
    
    for img_file in image_files:
        
        k_img = img_file[:-10]
        print(k_img)
        # Read image and convert into gray
        img = cv2.imread(os.path.join(data_dir,img_file))
        img_copy = img.copy()
        
        # Detecting columns and walls
        thresh = detect_walls(img)
        
        # Detecting lines
        gray = delete_walls(img)
        
        # Create default parametrization LSD
        lsd = cv2.createLineSegmentDetector(0)
        
        # Detect lines in the image
        dlines = lsd.detect(gray)
        lines = lsd.detect(gray)[0]
        
        # draw original lines
        # draw_ori_lines(img,lines)
        
        # preprocess lines
        Hlines, Vlines, dlinesDict = preprocess_lines(lines)
        
        # connect lines
        HLS,VLS,DLS = connect_all_lines(Hlines,Vlines,dlinesDict)
        
        # draw connected lines
        # draw_cnt_lines(img,HLS,VLS,DLS)
        
        # find intersection nodes
        Nodes, nodesDict,edgesDict = find_intersection_node(HLS,VLS,DLS)
        
        # delete duplicated nodes (nodesDict and edgesDict are merely edited in the 
        # following function, thus they donot need to be returned.)
        nodesDense = delete_dupli_nodes(Nodes,nodesDict,edgesDict)
        
        # draw nodes
        img_nodes,raw_nodes = draw_nodes(nodesDense,thresh)
                
        # establish graph
        Gstrc = establish_graph(nodesDense,HLS,VLS,DLS,raw_nodes)
        
        # export graph
        adj,nodeX = exportGraph(Gstrc,os.path.join(sheet_dir,k_img))
        
        # draw graph
        draw_graph(Gstrc,color ='#2F5597',file_name = os.path.join(graph_dir,k_img))