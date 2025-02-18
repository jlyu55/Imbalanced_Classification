library(Seurat)
library(SeuratData)
library(uwot)
library(ggrepel)
library(ggplot2)


#############################################
########### ifnb data
#############################################
# install dataset
InstallData("ifnb")

# load dataset
LoadData("ifnb")

# (ADD BY Jingyang Lyu)
###### solve an unexpected issue
ifnb <- UpdateSeuratObject(ifnb)

# split the dataset into a list 
data.list <- SplitObject(ifnb, split.by = "stim")

# normalize and identify variable features for each dataset independently
data.list <- lapply(X = data.list, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})

# select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = data.list, nfeatures = 2000)

table(data.list$CTRL@meta.data$seurat_annotations)
 #   CD14 Mono  CD4 Naive T CD4 Memory T    CD16 Mono            B        CD8 T 
 #        2215          978          859          507          407          352 
 # T activated           NK           DC  B Activated           Mk          pDC 
 #         300          298          258          185          115           51 
 #       Eryth 
 #          23 
table(data.list$STIM@meta.data$seurat_annotations)
 #   CD14 Mono  CD4 Naive T CD4 Memory T    CD16 Mono            B        CD8 T 
 #        2147         1526          903          537          571          462 
 # T activated           NK           DC  B Activated           Mk          pDC 
 #         333          321          214          203          121           81 
 #       Eryth 
 #          32 

data.X1 <- as.matrix(data.list$CTRL@assays$RNA@data)
ct.X1 <- data.list$CTRL@meta.data$seurat_annotations
data.X2 <- as.matrix(data.list$STIM@assays$RNA@data)
ct.X2 <- data.list$STIM@meta.data$seurat_annotations

########### feature selection

data.X1 <- data.X1[match(features,rownames(data.X1)), ]
data.X2 <- data.X2[match(features,rownames(data.X2)), ]
ct.X1 <- ct.X1[which(colSums(data.X1) != 0)]
ct.X2 <- ct.X2[which(colSums(data.X2) != 0)]
data.X1 <- data.X1[,which(colSums(data.X1) != 0)]
data.X2 <- data.X2[,which(colSums(data.X2) != 0)]

########### save data

setwd("../data")
save(data.X1, data.X2, ct.X1, ct.X2, file="ifnb.RData")

load("../data/ifnb.RData")
write.table(t(data.X1), file = "ifnb_control_X.txt")
write.table(ct.X1, file = "ifnb_control_y.txt")
write.table(t(data.X2), file = "ifnb_stimulated_X.txt")
write.table(ct.X2, file = "ifnb_stimulated_y.txt")
