ÿ¼1
ãÈ
:
Add
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
k
BatchMatMulV2
x"T
y"T
output"T"
Ttype:

2	"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
­
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisintÿÿÿÿÿÿÿÿÿ"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.14.02unknownÙà*
\
	input_idsPlaceholder*
dtype0*
_output_shapes
:	*
shape:	
]

input_maskPlaceholder*
_output_shapes
:	*
shape:	*
dtype0
^
segment_idsPlaceholder*
_output_shapes
:	*
shape:	*
dtype0
R
	label_idsPlaceholder*
dtype0*
_output_shapes
:*
shape:
q
bert/embeddings/ExpandDims/dimConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:

bert/embeddings/ExpandDims
ExpandDims	input_idsbert/embeddings/ExpandDims/dim*#
_output_shapes
:*

Tdim0*
T0
Ç
Bbert/embeddings/word_embeddings/Initializer/truncated_normal/shapeConst*
valueB":w     *2
_class(
&$loc:@bert/embeddings/word_embeddings*
dtype0*
_output_shapes
:
º
Abert/embeddings/word_embeddings/Initializer/truncated_normal/meanConst*
valueB
 *    *2
_class(
&$loc:@bert/embeddings/word_embeddings*
dtype0*
_output_shapes
: 
¼
Cbert/embeddings/word_embeddings/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*2
_class(
&$loc:@bert/embeddings/word_embeddings*
dtype0*
_output_shapes
: 
©
Lbert/embeddings/word_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalBbert/embeddings/word_embeddings/Initializer/truncated_normal/shape*2
_class(
&$loc:@bert/embeddings/word_embeddings*
seed2 *
dtype0*!
_output_shapes
:ºî*

seed *
T0
º
@bert/embeddings/word_embeddings/Initializer/truncated_normal/mulMulLbert/embeddings/word_embeddings/Initializer/truncated_normal/TruncatedNormalCbert/embeddings/word_embeddings/Initializer/truncated_normal/stddev*!
_output_shapes
:ºî*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings
¨
<bert/embeddings/word_embeddings/Initializer/truncated_normalAdd@bert/embeddings/word_embeddings/Initializer/truncated_normal/mulAbert/embeddings/word_embeddings/Initializer/truncated_normal/mean*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:ºî
Í
bert/embeddings/word_embeddings
VariableV2*
shape:ºî*
dtype0*!
_output_shapes
:ºî*
shared_name *2
_class(
&$loc:@bert/embeddings/word_embeddings*
	container 

&bert/embeddings/word_embeddings/AssignAssignbert/embeddings/word_embeddings<bert/embeddings/word_embeddings/Initializer/truncated_normal*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*
validate_shape(*!
_output_shapes
:ºî*
use_locking(
±
$bert/embeddings/word_embeddings/readIdentitybert/embeddings/word_embeddings*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:ºî*
T0
p
bert/embeddings/Reshape/shapeConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:

bert/embeddings/ReshapeReshapebert/embeddings/ExpandDimsbert/embeddings/Reshape/shape*
Tshape0*
_output_shapes	
:*
T0
_
bert/embeddings/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
â
bert/embeddings/GatherV2GatherV2$bert/embeddings/word_embeddings/readbert/embeddings/Reshapebert/embeddings/GatherV2/axis* 
_output_shapes
:
*
Taxis0*

batch_dims *
Tindices0*
Tparams0
t
bert/embeddings/Reshape_1/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:

bert/embeddings/Reshape_1Reshapebert/embeddings/GatherV2bert/embeddings/Reshape_1/shape*
T0*
Tshape0*$
_output_shapes
:
Ó
Hbert/embeddings/token_type_embeddings/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *8
_class.
,*loc:@bert/embeddings/token_type_embeddings
Æ
Gbert/embeddings/token_type_embeddings/Initializer/truncated_normal/meanConst*
valueB
 *    *8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
dtype0*
_output_shapes
: 
È
Ibert/embeddings/token_type_embeddings/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
dtype0*
_output_shapes
: 
¹
Rbert/embeddings/token_type_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHbert/embeddings/token_type_embeddings/Initializer/truncated_normal/shape*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
seed2 *
dtype0*
_output_shapes
:	*

seed *
T0
Ð
Fbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mulMulRbert/embeddings/token_type_embeddings/Initializer/truncated_normal/TruncatedNormalIbert/embeddings/token_type_embeddings/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	
¾
Bbert/embeddings/token_type_embeddings/Initializer/truncated_normalAddFbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mulGbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	
Õ
%bert/embeddings/token_type_embeddings
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
	container *
shape:	
®
,bert/embeddings/token_type_embeddings/AssignAssign%bert/embeddings/token_type_embeddingsBbert/embeddings/token_type_embeddings/Initializer/truncated_normal*
_output_shapes
:	*
use_locking(*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
validate_shape(
Á
*bert/embeddings/token_type_embeddings/readIdentity%bert/embeddings/token_type_embeddings*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	
r
bert/embeddings/Reshape_2/shapeConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:

bert/embeddings/Reshape_2Reshapesegment_idsbert/embeddings/Reshape_2/shape*
T0*
Tshape0*
_output_shapes	
:
e
 bert/embeddings/one_hot/on_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
f
!bert/embeddings/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
_
bert/embeddings/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
é
bert/embeddings/one_hotOneHotbert/embeddings/Reshape_2bert/embeddings/one_hot/depth bert/embeddings/one_hot/on_value!bert/embeddings/one_hot/off_value*
_output_shapes
:	*
T0*
axisÿÿÿÿÿÿÿÿÿ*
TI0
¶
bert/embeddings/MatMulMatMulbert/embeddings/one_hot*bert/embeddings/token_type_embeddings/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
t
bert/embeddings/Reshape_3/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:

bert/embeddings/Reshape_3Reshapebert/embeddings/MatMulbert/embeddings/Reshape_3/shape*$
_output_shapes
:*
T0*
Tshape0

bert/embeddings/addAddbert/embeddings/Reshape_1bert/embeddings/Reshape_3*
T0*$
_output_shapes
:
f
#bert/embeddings/assert_less_equal/xConst*
value
B :*
dtype0*
_output_shapes
: 
f
#bert/embeddings/assert_less_equal/yConst*
value
B :*
dtype0*
_output_shapes
: 
£
+bert/embeddings/assert_less_equal/LessEqual	LessEqual#bert/embeddings/assert_less_equal/x#bert/embeddings/assert_less_equal/y*
T0*
_output_shapes
: 
j
'bert/embeddings/assert_less_equal/ConstConst*
dtype0*
_output_shapes
: *
valueB 
·
%bert/embeddings/assert_less_equal/AllAll+bert/embeddings/assert_less_equal/LessEqual'bert/embeddings/assert_less_equal/Const*
	keep_dims( *

Tidx0*
_output_shapes
: 
o
.bert/embeddings/assert_less_equal/Assert/ConstConst*
valueB B *
dtype0*
_output_shapes
: 
È
0bert/embeddings/assert_less_equal/Assert/Const_1Const*
dtype0*
_output_shapes
: *h
value_B] BWCondition x <= y did not hold element-wise:x (bert/embeddings/assert_less_equal/x:0) = 

0bert/embeddings/assert_less_equal/Assert/Const_2Const*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = *
dtype0*
_output_shapes
: 
w
6bert/embeddings/assert_less_equal/Assert/Assert/data_0Const*
valueB B *
dtype0*
_output_shapes
: 
Î
6bert/embeddings/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *h
value_B] BWCondition x <= y did not hold element-wise:x (bert/embeddings/assert_less_equal/x:0) = *
dtype0
£
6bert/embeddings/assert_less_equal/Assert/Assert/data_3Const*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = *
dtype0*
_output_shapes
: 
ó
/bert/embeddings/assert_less_equal/Assert/AssertAssert%bert/embeddings/assert_less_equal/All6bert/embeddings/assert_less_equal/Assert/Assert/data_06bert/embeddings/assert_less_equal/Assert/Assert/data_1#bert/embeddings/assert_less_equal/x6bert/embeddings/assert_less_equal/Assert/Assert/data_3#bert/embeddings/assert_less_equal/y*
T	
2*
	summarize
Ï
Fbert/embeddings/position_embeddings/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *6
_class,
*(loc:@bert/embeddings/position_embeddings*
dtype0
Â
Ebert/embeddings/position_embeddings/Initializer/truncated_normal/meanConst*
valueB
 *    *6
_class,
*(loc:@bert/embeddings/position_embeddings*
dtype0*
_output_shapes
: 
Ä
Gbert/embeddings/position_embeddings/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*6
_class,
*(loc:@bert/embeddings/position_embeddings*
dtype0*
_output_shapes
: 
´
Pbert/embeddings/position_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFbert/embeddings/position_embeddings/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings*
seed2 
É
Dbert/embeddings/position_embeddings/Initializer/truncated_normal/mulMulPbert/embeddings/position_embeddings/Initializer/truncated_normal/TruncatedNormalGbert/embeddings/position_embeddings/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings
·
@bert/embeddings/position_embeddings/Initializer/truncated_normalAddDbert/embeddings/position_embeddings/Initializer/truncated_normal/mulEbert/embeddings/position_embeddings/Initializer/truncated_normal/mean*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:

Ó
#bert/embeddings/position_embeddings
VariableV2*
shared_name *6
_class,
*(loc:@bert/embeddings/position_embeddings*
	container *
shape:
*
dtype0* 
_output_shapes
:

§
*bert/embeddings/position_embeddings/AssignAssign#bert/embeddings/position_embeddings@bert/embeddings/position_embeddings/Initializer/truncated_normal*6
_class,
*(loc:@bert/embeddings/position_embeddings*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
¼
(bert/embeddings/position_embeddings/readIdentity#bert/embeddings/position_embeddings*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
*
T0

bert/embeddings/Slice/beginConst0^bert/embeddings/assert_less_equal/Assert/Assert*
valueB"        *
dtype0*
_output_shapes
:

bert/embeddings/Slice/sizeConst0^bert/embeddings/assert_less_equal/Assert/Assert*
valueB"   ÿÿÿÿ*
dtype0*
_output_shapes
:
¹
bert/embeddings/SliceSlice(bert/embeddings/position_embeddings/readbert/embeddings/Slice/beginbert/embeddings/Slice/size*
Index0*
T0* 
_output_shapes
:

¦
bert/embeddings/Reshape_4/shapeConst0^bert/embeddings/assert_less_equal/Assert/Assert*!
valueB"         *
dtype0*
_output_shapes
:

bert/embeddings/Reshape_4Reshapebert/embeddings/Slicebert/embeddings/Reshape_4/shape*
T0*
Tshape0*$
_output_shapes
:
{
bert/embeddings/add_1Addbert/embeddings/addbert/embeddings/Reshape_4*$
_output_shapes
:*
T0
²
0bert/embeddings/LayerNorm/beta/Initializer/zerosConst*
valueB*    *1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
dtype0*
_output_shapes	
:
¿
bert/embeddings/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
	container *
shape:

%bert/embeddings/LayerNorm/beta/AssignAssignbert/embeddings/LayerNorm/beta0bert/embeddings/LayerNorm/beta/Initializer/zeros*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
¨
#bert/embeddings/LayerNorm/beta/readIdentitybert/embeddings/LayerNorm/beta*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
_output_shapes	
:
³
0bert/embeddings/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Á
bert/embeddings/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
	container *
shape:

&bert/embeddings/LayerNorm/gamma/AssignAssignbert/embeddings/LayerNorm/gamma0bert/embeddings/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma
«
$bert/embeddings/LayerNorm/gamma/readIdentitybert/embeddings/LayerNorm/gamma*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
_output_shapes	
:

8bert/embeddings/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
Ê
&bert/embeddings/LayerNorm/moments/meanMeanbert/embeddings/add_18bert/embeddings/LayerNorm/moments/mean/reduction_indices*#
_output_shapes
:*
	keep_dims(*

Tidx0*
T0

.bert/embeddings/LayerNorm/moments/StopGradientStopGradient&bert/embeddings/LayerNorm/moments/mean*
T0*#
_output_shapes
:
¾
3bert/embeddings/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/embeddings/add_1.bert/embeddings/LayerNorm/moments/StopGradient*
T0*$
_output_shapes
:

<bert/embeddings/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ð
*bert/embeddings/LayerNorm/moments/varianceMean3bert/embeddings/LayerNorm/moments/SquaredDifference<bert/embeddings/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*#
_output_shapes
:
n
)bert/embeddings/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
³
'bert/embeddings/LayerNorm/batchnorm/addAdd*bert/embeddings/LayerNorm/moments/variance)bert/embeddings/LayerNorm/batchnorm/add/y*#
_output_shapes
:*
T0

)bert/embeddings/LayerNorm/batchnorm/RsqrtRsqrt'bert/embeddings/LayerNorm/batchnorm/add*
T0*#
_output_shapes
:
®
'bert/embeddings/LayerNorm/batchnorm/mulMul)bert/embeddings/LayerNorm/batchnorm/Rsqrt$bert/embeddings/LayerNorm/gamma/read*$
_output_shapes
:*
T0

)bert/embeddings/LayerNorm/batchnorm/mul_1Mulbert/embeddings/add_1'bert/embeddings/LayerNorm/batchnorm/mul*$
_output_shapes
:*
T0
°
)bert/embeddings/LayerNorm/batchnorm/mul_2Mul&bert/embeddings/LayerNorm/moments/mean'bert/embeddings/LayerNorm/batchnorm/mul*$
_output_shapes
:*
T0
­
'bert/embeddings/LayerNorm/batchnorm/subSub#bert/embeddings/LayerNorm/beta/read)bert/embeddings/LayerNorm/batchnorm/mul_2*
T0*$
_output_shapes
:
³
)bert/embeddings/LayerNorm/batchnorm/add_1Add)bert/embeddings/LayerNorm/batchnorm/mul_1'bert/embeddings/LayerNorm/batchnorm/sub*
T0*$
_output_shapes
:
o
bert/encoder/Reshape/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:

bert/encoder/ReshapeReshape
input_maskbert/encoder/Reshape/shape*#
_output_shapes
:*
T0*
Tshape0
|
bert/encoder/CastCastbert/encoder/Reshape*

SrcT0*
Truncate( *#
_output_shapes
:*

DstT0
p
bert/encoder/onesConst*
dtype0*#
_output_shapes
:*"
valueB*  ?
l
bert/encoder/mulMulbert/encoder/onesbert/encoder/Cast*$
_output_shapes
:*
T0
m
bert/encoder/Reshape_1/shapeConst*
valueB"ÿÿÿÿ   *
dtype0*
_output_shapes
:
£
bert/encoder/Reshape_1Reshape)bert/embeddings/LayerNorm/batchnorm/add_1bert/encoder/Reshape_1/shape*
T0*
Tshape0* 
_output_shapes
:

é
Sbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel
Þ
Tbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel
ý
Qbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/stddev*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:
*
T0
ë
Mbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:

í
0bert/encoder/layer_0/attention/self/query/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
	container *
shape:

Û
7bert/encoder/layer_0/attention/self/query/kernel/AssignAssign0bert/encoder/layer_0/attention/self/query/kernelMbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ã
5bert/encoder/layer_0/attention/self/query/kernel/readIdentity0bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel
Ò
@bert/encoder/layer_0/attention/self/query/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_0/attention/self/query/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias
Ã
5bert/encoder/layer_0/attention/self/query/bias/AssignAssign.bert/encoder/layer_0/attention/self/query/bias@bert/encoder/layer_0/attention/self/query/bias/Initializer/zeros*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ø
3bert/encoder/layer_0/attention/self/query/bias/readIdentity.bert/encoder/layer_0/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
_output_shapes	
:
Ú
0bert/encoder/layer_0/attention/self/query/MatMulMatMulbert/encoder/Reshape_15bert/encoder/layer_0/attention/self/query/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
å
1bert/encoder/layer_0/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_0/attention/self/query/MatMul3bert/encoder/layer_0/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

å
Qbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/shape*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0
õ
Obert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel
é
.bert/encoder/layer_0/attention/self/key/kernel
VariableV2*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ó
5bert/encoder/layer_0/attention/self/key/kernel/AssignAssign.bert/encoder/layer_0/attention/self/key/kernelKbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
validate_shape(
Ý
3bert/encoder/layer_0/attention/self/key/kernel/readIdentity.bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel
Î
>bert/encoder/layer_0/attention/self/key/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias
Û
,bert/encoder/layer_0/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
	container *
shape:
»
3bert/encoder/layer_0/attention/self/key/bias/AssignAssign,bert/encoder/layer_0/attention/self/key/bias>bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ò
1bert/encoder/layer_0/attention/self/key/bias/readIdentity,bert/encoder/layer_0/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
_output_shapes	
:
Ö
.bert/encoder/layer_0/attention/self/key/MatMulMatMulbert/encoder/Reshape_13bert/encoder/layer_0/attention/self/key/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ß
/bert/encoder/layer_0/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_0/attention/self/key/MatMul1bert/encoder/layer_0/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

é
Sbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
seed2 *
dtype0
ý
Qbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel
ë
Mbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:

í
0bert/encoder/layer_0/attention/self/value/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Û
7bert/encoder/layer_0/attention/self/value/kernel/AssignAssign0bert/encoder/layer_0/attention/self/value/kernelMbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ã
5bert/encoder/layer_0/attention/self/value/kernel/readIdentity0bert/encoder/layer_0/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_0/attention/self/value/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_0/attention/self/value/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ã
5bert/encoder/layer_0/attention/self/value/bias/AssignAssign.bert/encoder/layer_0/attention/self/value/bias@bert/encoder/layer_0/attention/self/value/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_0/attention/self/value/bias/readIdentity.bert/encoder/layer_0/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
_output_shapes	
:
Ú
0bert/encoder/layer_0/attention/self/value/MatMulMatMulbert/encoder/Reshape_15bert/encoder/layer_0/attention/self/value/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_0/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_0/attention/self/value/MatMul3bert/encoder/layer_0/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0

1bert/encoder/layer_0/attention/self/Reshape/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
Ü
+bert/encoder/layer_0/attention/self/ReshapeReshape1bert/encoder/layer_0/attention/self/query/BiasAdd1bert/encoder/layer_0/attention/self/Reshape/shape*
Tshape0*'
_output_shapes
:@*
T0

2bert/encoder/layer_0/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_0/attention/self/transpose	Transpose+bert/encoder/layer_0/attention/self/Reshape2bert/encoder/layer_0/attention/self/transpose/perm*
T0*'
_output_shapes
:@*
Tperm0

3bert/encoder/layer_0/attention/self/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Þ
-bert/encoder/layer_0/attention/self/Reshape_1Reshape/bert/encoder/layer_0/attention/self/key/BiasAdd3bert/encoder/layer_0/attention/self/Reshape_1/shape*
Tshape0*'
_output_shapes
:@*
T0

4bert/encoder/layer_0/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_0/attention/self/transpose_1	Transpose-bert/encoder/layer_0/attention/self/Reshape_14bert/encoder/layer_0/attention/self/transpose_1/perm*
T0*'
_output_shapes
:@*
Tperm0
è
*bert/encoder/layer_0/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_0/attention/self/transpose/bert/encoder/layer_0/attention/self/transpose_1*(
_output_shapes
:*
adj_x( *
adj_y(*
T0
n
)bert/encoder/layer_0/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
¸
'bert/encoder/layer_0/attention/self/MulMul*bert/encoder/layer_0/attention/self/MatMul)bert/encoder/layer_0/attention/self/Mul/y*
T0*(
_output_shapes
:
|
2bert/encoder/layer_0/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Á
.bert/encoder/layer_0/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_0/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:
n
)bert/encoder/layer_0/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¼
'bert/encoder/layer_0/attention/self/subSub)bert/encoder/layer_0/attention/self/sub/x.bert/encoder/layer_0/attention/self/ExpandDims*
T0*(
_output_shapes
:
p
+bert/encoder/layer_0/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¹
)bert/encoder/layer_0/attention/self/mul_1Mul'bert/encoder/layer_0/attention/self/sub+bert/encoder/layer_0/attention/self/mul_1/y*(
_output_shapes
:*
T0
µ
'bert/encoder/layer_0/attention/self/addAdd'bert/encoder/layer_0/attention/self/Mul)bert/encoder/layer_0/attention/self/mul_1*(
_output_shapes
:*
T0

+bert/encoder/layer_0/attention/self/SoftmaxSoftmax'bert/encoder/layer_0/attention/self/add*
T0*(
_output_shapes
:

3bert/encoder/layer_0/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
à
-bert/encoder/layer_0/attention/self/Reshape_2Reshape1bert/encoder/layer_0/attention/self/value/BiasAdd3bert/encoder/layer_0/attention/self/Reshape_2/shape*
Tshape0*'
_output_shapes
:@*
T0

4bert/encoder/layer_0/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_0/attention/self/transpose_2	Transpose-bert/encoder/layer_0/attention/self/Reshape_24bert/encoder/layer_0/attention/self/transpose_2/perm*
T0*'
_output_shapes
:@*
Tperm0
ç
,bert/encoder/layer_0/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_0/attention/self/Softmax/bert/encoder/layer_0/attention/self/transpose_2*'
_output_shapes
:@*
adj_x( *
adj_y( *
T0

4bert/encoder/layer_0/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ß
/bert/encoder/layer_0/attention/self/transpose_3	Transpose,bert/encoder/layer_0/attention/self/MatMul_14bert/encoder/layer_0/attention/self/transpose_3/perm*
Tperm0*
T0*'
_output_shapes
:@

3bert/encoder/layer_0/attention/self/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
×
-bert/encoder/layer_0/attention/self/Reshape_3Reshape/bert/encoder/layer_0/attention/self/transpose_33bert/encoder/layer_0/attention/self/Reshape_3/shape* 
_output_shapes
:
*
T0*
Tshape0
í
Ubert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel
à
Tbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
â
Vbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
á
_bert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
seed2 

Sbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel
ñ
2bert/encoder/layer_0/attention/output/dense/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel
ã
9bert/encoder/layer_0/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_0/attention/output/dense/kernelObert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

é
7bert/encoder/layer_0/attention/output/dense/kernel/readIdentity2bert/encoder/layer_0/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:

Ö
Bbert/encoder/layer_0/attention/output/dense/bias/Initializer/zerosConst*
valueB*    *C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
dtype0*
_output_shapes	
:
ã
0bert/encoder/layer_0/attention/output/dense/bias
VariableV2*
_output_shapes	
:*
shared_name *C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
	container *
shape:*
dtype0
Ë
7bert/encoder/layer_0/attention/output/dense/bias/AssignAssign0bert/encoder/layer_0/attention/output/dense/biasBbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
Þ
5bert/encoder/layer_0/attention/output/dense/bias/readIdentity0bert/encoder/layer_0/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
_output_shapes	
:
õ
2bert/encoder/layer_0/attention/output/dense/MatMulMatMul-bert/encoder/layer_0/attention/self/Reshape_37bert/encoder/layer_0/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ë
3bert/encoder/layer_0/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_0/attention/output/dense/MatMul5bert/encoder/layer_0/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

¨
)bert/encoder/layer_0/attention/output/addAdd3bert/encoder/layer_0/attention/output/dense/BiasAddbert/encoder/Reshape_1* 
_output_shapes
:
*
T0
Þ
Fbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
ë
4bert/encoder/layer_0/attention/output/LayerNorm/beta
VariableV2*
shared_name *G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
Û
;bert/encoder/layer_0/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_0/attention/output/LayerNorm/betaFbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ê
9bert/encoder/layer_0/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_0/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
_output_shapes	
:
ß
Fbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_0/attention/output/LayerNorm/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
	container 
Þ
<bert/encoder/layer_0/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_0/attention/output/LayerNorm/gammaFbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
í
:bert/encoder/layer_0/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_0/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
_output_shapes	
:

Nbert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

<bert/encoder/layer_0/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_0/attention/output/addNbert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¼
Dbert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_0/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ú
Ibert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_0/attention/output/addDbert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
*
T0

Rbert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_0/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0

?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_0/attention/output/LayerNorm/moments/variance?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	
±
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ì
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_0/attention/output/LayerNorm/gamma/read* 
_output_shapes
:
*
T0
Û
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_0/attention/output/add=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_0/attention/output/LayerNorm/moments/mean=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

ë
=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_0/attention/output/LayerNorm/beta/read?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
*
T0
ñ
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
å
Qbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel
Ø
Pbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel
õ
Obert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel
é
.bert/encoder/layer_0/intermediate/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
	container *
shape:

Ó
5bert/encoder/layer_0/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_0/intermediate/dense/kernelKbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_0/intermediate/dense/kernel/readIdentity.bert/encoder/layer_0/intermediate/dense/kernel*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:
*
T0
Ú
Nbert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB:*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
dtype0
Ê
Dbert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
dtype0*
_output_shapes
: 
Õ
>bert/encoder/layer_0/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias
Û
,bert/encoder/layer_0/intermediate/dense/bias
VariableV2*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
»
3bert/encoder/layer_0/intermediate/dense/bias/AssignAssign,bert/encoder/layer_0/intermediate/dense/bias>bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_0/intermediate/dense/bias/readIdentity,bert/encoder/layer_0/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
_output_shapes	
:
ÿ
.bert/encoder/layer_0/intermediate/dense/MatMulMatMul?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_0/intermediate/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ß
/bert/encoder/layer_0/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_0/intermediate/dense/MatMul1bert/encoder/layer_0/intermediate/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
r
-bert/encoder/layer_0/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
½
+bert/encoder/layer_0/intermediate/dense/PowPow/bert/encoder/layer_0/intermediate/dense/BiasAdd-bert/encoder/layer_0/intermediate/dense/Pow/y*
T0* 
_output_shapes
:

r
-bert/encoder/layer_0/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
¹
+bert/encoder/layer_0/intermediate/dense/mulMul-bert/encoder/layer_0/intermediate/dense/mul/x+bert/encoder/layer_0/intermediate/dense/Pow*
T0* 
_output_shapes
:

»
+bert/encoder/layer_0/intermediate/dense/addAdd/bert/encoder/layer_0/intermediate/dense/BiasAdd+bert/encoder/layer_0/intermediate/dense/mul*
T0* 
_output_shapes
:

t
/bert/encoder/layer_0/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
½
-bert/encoder/layer_0/intermediate/dense/mul_1Mul/bert/encoder/layer_0/intermediate/dense/mul_1/x+bert/encoder/layer_0/intermediate/dense/add* 
_output_shapes
:
*
T0

,bert/encoder/layer_0/intermediate/dense/TanhTanh-bert/encoder/layer_0/intermediate/dense/mul_1*
T0* 
_output_shapes
:

t
/bert/encoder/layer_0/intermediate/dense/add_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
¾
-bert/encoder/layer_0/intermediate/dense/add_1Add/bert/encoder/layer_0/intermediate/dense/add_1/x,bert/encoder/layer_0/intermediate/dense/Tanh* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_0/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
¿
-bert/encoder/layer_0/intermediate/dense/mul_2Mul/bert/encoder/layer_0/intermediate/dense/mul_2/x-bert/encoder/layer_0/intermediate/dense/add_1* 
_output_shapes
:
*
T0
¿
-bert/encoder/layer_0/intermediate/dense/mul_3Mul/bert/encoder/layer_0/intermediate/dense/BiasAdd-bert/encoder/layer_0/intermediate/dense/mul_2*
T0* 
_output_shapes
:

Ù
Kbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel
Î
Lbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/shape*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ý
Ibert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:

Ý
(bert/encoder/layer_0/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
	container *
shape:

»
/bert/encoder/layer_0/output/dense/kernel/AssignAssign(bert/encoder/layer_0/output/dense/kernelEbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ë
-bert/encoder/layer_0/output/dense/kernel/readIdentity(bert/encoder/layer_0/output/dense/kernel*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:
*
T0
Â
8bert/encoder/layer_0/output/dense/bias/Initializer/zerosConst*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
dtype0*
_output_shapes	
:
Ï
&bert/encoder/layer_0/output/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
	container 
£
-bert/encoder/layer_0/output/dense/bias/AssignAssign&bert/encoder/layer_0/output/dense/bias8bert/encoder/layer_0/output/dense/bias/Initializer/zeros*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
À
+bert/encoder/layer_0/output/dense/bias/readIdentity&bert/encoder/layer_0/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
_output_shapes	
:
á
(bert/encoder/layer_0/output/dense/MatMulMatMul-bert/encoder/layer_0/intermediate/dense/mul_3-bert/encoder/layer_0/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
Í
)bert/encoder/layer_0/output/dense/BiasAddBiasAdd(bert/encoder/layer_0/output/dense/MatMul+bert/encoder/layer_0/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

½
bert/encoder/layer_0/output/addAdd)bert/encoder/layer_0/output/dense/BiasAdd?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ê
<bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_0/output/LayerNorm/beta
VariableV2*
shared_name *=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
³
1bert/encoder/layer_0/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_0/output/LayerNorm/beta<bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ì
/bert/encoder/layer_0/output/LayerNorm/beta/readIdentity*bert/encoder/layer_0/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
_output_shapes	
:
Ë
<bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Ù
+bert/encoder/layer_0/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
	container *
shape:
¶
2bert/encoder/layer_0/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_0/output/LayerNorm/gamma<bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_0/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_0/output/LayerNorm/gamma*
_output_shapes	
:*
T0*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma

Dbert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
è
2bert/encoder/layer_0/output/LayerNorm/moments/meanMeanbert/encoder/layer_0/output/addDbert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¨
:bert/encoder/layer_0/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_0/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_0/output/add:bert/encoder/layer_0/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Hbert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0

6bert/encoder/layer_0/output/LayerNorm/moments/varianceMean?bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
z
5bert/encoder/layer_0/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_0/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_0/output/LayerNorm/moments/variance5bert/encoder/layer_0/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0

5bert/encoder/layer_0/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_0/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
Î
3bert/encoder/layer_0/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_0/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

½
5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_0/output/add3bert/encoder/layer_0/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_0/output/LayerNorm/moments/mean3bert/encoder/layer_0/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Í
3bert/encoder/layer_0/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_0/output/LayerNorm/beta/read5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
*
T0
Ó
5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_0/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

é
Sbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
dtype0
Û
]bert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
seed2 
ý
Qbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel
ë
Mbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:

í
0bert/encoder/layer_1/attention/self/query/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Û
7bert/encoder/layer_1/attention/self/query/kernel/AssignAssign0bert/encoder/layer_1/attention/self/query/kernelMbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ã
5bert/encoder/layer_1/attention/self/query/kernel/readIdentity0bert/encoder/layer_1/attention/self/query/kernel*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:
*
T0
Ò
@bert/encoder/layer_1/attention/self/query/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias
ß
.bert/encoder/layer_1/attention/self/query/bias
VariableV2*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
	container *
shape:*
dtype0
Ã
5bert/encoder/layer_1/attention/self/query/bias/AssignAssign.bert/encoder/layer_1/attention/self/query/bias@bert/encoder/layer_1/attention/self/query/bias/Initializer/zeros*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ø
3bert/encoder/layer_1/attention/self/query/bias/readIdentity.bert/encoder/layer_1/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
_output_shapes	
:
ù
0bert/encoder/layer_1/attention/self/query/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_15bert/encoder/layer_1/attention/self/query/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
å
1bert/encoder/layer_1/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_1/attention/self/query/MatMul3bert/encoder/layer_1/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

å
Qbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
seed2 
õ
Obert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel
é
.bert/encoder/layer_1/attention/self/key/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
	container *
shape:

Ó
5bert/encoder/layer_1/attention/self/key/kernel/AssignAssign.bert/encoder/layer_1/attention/self/key/kernelKbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel
Ý
3bert/encoder/layer_1/attention/self/key/kernel/readIdentity.bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel
Î
>bert/encoder/layer_1/attention/self/key/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
dtype0
Û
,bert/encoder/layer_1/attention/self/key/bias
VariableV2*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
	container *
shape:*
dtype0
»
3bert/encoder/layer_1/attention/self/key/bias/AssignAssign,bert/encoder/layer_1/attention/self/key/bias>bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_1/attention/self/key/bias/readIdentity,bert/encoder/layer_1/attention/self/key/bias*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
_output_shapes	
:*
T0
õ
.bert/encoder/layer_1/attention/self/key/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_13bert/encoder/layer_1/attention/self/key/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ß
/bert/encoder/layer_1/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_1/attention/self/key/MatMul1bert/encoder/layer_1/attention/self/key/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
é
Sbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel
Ü
Rbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ý
Qbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mean*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:
*
T0
í
0bert/encoder/layer_1/attention/self/value/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
	container *
shape:

Û
7bert/encoder/layer_1/attention/self/value/kernel/AssignAssign0bert/encoder/layer_1/attention/self/value/kernelMbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_1/attention/self/value/kernel/readIdentity0bert/encoder/layer_1/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_1/attention/self/value/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_1/attention/self/value/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
	container 
Ã
5bert/encoder/layer_1/attention/self/value/bias/AssignAssign.bert/encoder/layer_1/attention/self/value/bias@bert/encoder/layer_1/attention/self/value/bias/Initializer/zeros*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ø
3bert/encoder/layer_1/attention/self/value/bias/readIdentity.bert/encoder/layer_1/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
_output_shapes	
:
ù
0bert/encoder/layer_1/attention/self/value/MatMulMatMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_15bert/encoder/layer_1/attention/self/value/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
å
1bert/encoder/layer_1/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_1/attention/self/value/MatMul3bert/encoder/layer_1/attention/self/value/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC

1bert/encoder/layer_1/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Ü
+bert/encoder/layer_1/attention/self/ReshapeReshape1bert/encoder/layer_1/attention/self/query/BiasAdd1bert/encoder/layer_1/attention/self/Reshape/shape*'
_output_shapes
:@*
T0*
Tshape0

2bert/encoder/layer_1/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_1/attention/self/transpose	Transpose+bert/encoder/layer_1/attention/self/Reshape2bert/encoder/layer_1/attention/self/transpose/perm*
T0*'
_output_shapes
:@*
Tperm0

3bert/encoder/layer_1/attention/self/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
Þ
-bert/encoder/layer_1/attention/self/Reshape_1Reshape/bert/encoder/layer_1/attention/self/key/BiasAdd3bert/encoder/layer_1/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_1/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_1/attention/self/transpose_1	Transpose-bert/encoder/layer_1/attention/self/Reshape_14bert/encoder/layer_1/attention/self/transpose_1/perm*'
_output_shapes
:@*
Tperm0*
T0
è
*bert/encoder/layer_1/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_1/attention/self/transpose/bert/encoder/layer_1/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:
n
)bert/encoder/layer_1/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
¸
'bert/encoder/layer_1/attention/self/MulMul*bert/encoder/layer_1/attention/self/MatMul)bert/encoder/layer_1/attention/self/Mul/y*(
_output_shapes
:*
T0
|
2bert/encoder/layer_1/attention/self/ExpandDims/dimConst*
_output_shapes
:*
valueB:*
dtype0
Á
.bert/encoder/layer_1/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_1/attention/self/ExpandDims/dim*
T0*(
_output_shapes
:*

Tdim0
n
)bert/encoder/layer_1/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¼
'bert/encoder/layer_1/attention/self/subSub)bert/encoder/layer_1/attention/self/sub/x.bert/encoder/layer_1/attention/self/ExpandDims*
T0*(
_output_shapes
:
p
+bert/encoder/layer_1/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¹
)bert/encoder/layer_1/attention/self/mul_1Mul'bert/encoder/layer_1/attention/self/sub+bert/encoder/layer_1/attention/self/mul_1/y*
T0*(
_output_shapes
:
µ
'bert/encoder/layer_1/attention/self/addAdd'bert/encoder/layer_1/attention/self/Mul)bert/encoder/layer_1/attention/self/mul_1*
T0*(
_output_shapes
:

+bert/encoder/layer_1/attention/self/SoftmaxSoftmax'bert/encoder/layer_1/attention/self/add*(
_output_shapes
:*
T0

3bert/encoder/layer_1/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
à
-bert/encoder/layer_1/attention/self/Reshape_2Reshape1bert/encoder/layer_1/attention/self/value/BiasAdd3bert/encoder/layer_1/attention/self/Reshape_2/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_1/attention/self/transpose_2/permConst*
_output_shapes
:*%
valueB"             *
dtype0
à
/bert/encoder/layer_1/attention/self/transpose_2	Transpose-bert/encoder/layer_1/attention/self/Reshape_24bert/encoder/layer_1/attention/self/transpose_2/perm*
T0*'
_output_shapes
:@*
Tperm0
ç
,bert/encoder/layer_1/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_1/attention/self/Softmax/bert/encoder/layer_1/attention/self/transpose_2*
adj_y( *
T0*'
_output_shapes
:@*
adj_x( 

4bert/encoder/layer_1/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ß
/bert/encoder/layer_1/attention/self/transpose_3	Transpose,bert/encoder/layer_1/attention/self/MatMul_14bert/encoder/layer_1/attention/self/transpose_3/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_1/attention/self/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
×
-bert/encoder/layer_1/attention/self/Reshape_3Reshape/bert/encoder/layer_1/attention/self/transpose_33bert/encoder/layer_1/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:

í
Ubert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
dtype0*
_output_shapes
:
à
Tbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
â
Vbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
á
_bert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
seed2 

Sbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/stddev*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:
*
T0
ó
Obert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:

ñ
2bert/encoder/layer_1/attention/output/dense/kernel
VariableV2*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
ã
9bert/encoder/layer_1/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_1/attention/output/dense/kernelObert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
validate_shape(
é
7bert/encoder/layer_1/attention/output/dense/kernel/readIdentity2bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel
Ö
Bbert/encoder/layer_1/attention/output/dense/bias/Initializer/zerosConst*
valueB*    *C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
dtype0*
_output_shapes	
:
ã
0bert/encoder/layer_1/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
	container *
shape:
Ë
7bert/encoder/layer_1/attention/output/dense/bias/AssignAssign0bert/encoder/layer_1/attention/output/dense/biasBbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Þ
5bert/encoder/layer_1/attention/output/dense/bias/readIdentity0bert/encoder/layer_1/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
_output_shapes	
:
õ
2bert/encoder/layer_1/attention/output/dense/MatMulMatMul-bert/encoder/layer_1/attention/self/Reshape_37bert/encoder/layer_1/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ë
3bert/encoder/layer_1/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_1/attention/output/dense/MatMul5bert/encoder/layer_1/attention/output/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
Ç
)bert/encoder/layer_1/attention/output/addAdd3bert/encoder/layer_1/attention/output/dense/BiasAdd5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Þ
Fbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
ë
4bert/encoder/layer_1/attention/output/LayerNorm/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
	container 
Û
;bert/encoder/layer_1/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_1/attention/output/LayerNorm/betaFbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ê
9bert/encoder/layer_1/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_1/attention/output/LayerNorm/beta*
_output_shapes	
:*
T0*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta
ß
Fbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_1/attention/output/LayerNorm/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma
Þ
<bert/encoder/layer_1/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_1/attention/output/LayerNorm/gammaFbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma
í
:bert/encoder/layer_1/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_1/attention/output/LayerNorm/gamma*
_output_shapes	
:*
T0*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma

Nbert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

<bert/encoder/layer_1/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_1/attention/output/addNbert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¼
Dbert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_1/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ú
Ibert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_1/attention/output/addDbert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
*
T0

Rbert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_1/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0

?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *Ì¼+
ñ
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_1/attention/output/LayerNorm/moments/variance?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	
±
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ì
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_1/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Û
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_1/attention/output/add=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_1/attention/output/LayerNorm/moments/mean=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

ë
=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_1/attention/output/LayerNorm/beta/read?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ñ
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

å
Qbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel
Ú
Rbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:

õ
Obert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel
é
.bert/encoder/layer_1/intermediate/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
	container *
shape:

Ó
5bert/encoder/layer_1/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_1/intermediate/dense/kernelKbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_1/intermediate/dense/kernel/readIdentity.bert/encoder/layer_1/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:

Ú
Nbert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ê
Dbert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
dtype0*
_output_shapes
: 
Õ
>bert/encoder/layer_1/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros/Const*
T0*

index_type0*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
_output_shapes	
:
Û
,bert/encoder/layer_1/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
	container *
shape:
»
3bert/encoder/layer_1/intermediate/dense/bias/AssignAssign,bert/encoder/layer_1/intermediate/dense/bias>bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_1/intermediate/dense/bias/readIdentity,bert/encoder/layer_1/intermediate/dense/bias*
_output_shapes	
:*
T0*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias
ÿ
.bert/encoder/layer_1/intermediate/dense/MatMulMatMul?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_1/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ß
/bert/encoder/layer_1/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_1/intermediate/dense/MatMul1bert/encoder/layer_1/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_1/intermediate/dense/Pow/yConst*
_output_shapes
: *
valueB
 *  @@*
dtype0
½
+bert/encoder/layer_1/intermediate/dense/PowPow/bert/encoder/layer_1/intermediate/dense/BiasAdd-bert/encoder/layer_1/intermediate/dense/Pow/y* 
_output_shapes
:
*
T0
r
-bert/encoder/layer_1/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
¹
+bert/encoder/layer_1/intermediate/dense/mulMul-bert/encoder/layer_1/intermediate/dense/mul/x+bert/encoder/layer_1/intermediate/dense/Pow*
T0* 
_output_shapes
:

»
+bert/encoder/layer_1/intermediate/dense/addAdd/bert/encoder/layer_1/intermediate/dense/BiasAdd+bert/encoder/layer_1/intermediate/dense/mul* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_1/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
½
-bert/encoder/layer_1/intermediate/dense/mul_1Mul/bert/encoder/layer_1/intermediate/dense/mul_1/x+bert/encoder/layer_1/intermediate/dense/add* 
_output_shapes
:
*
T0

,bert/encoder/layer_1/intermediate/dense/TanhTanh-bert/encoder/layer_1/intermediate/dense/mul_1* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_1/intermediate/dense/add_1/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
¾
-bert/encoder/layer_1/intermediate/dense/add_1Add/bert/encoder/layer_1/intermediate/dense/add_1/x,bert/encoder/layer_1/intermediate/dense/Tanh*
T0* 
_output_shapes
:

t
/bert/encoder/layer_1/intermediate/dense/mul_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
¿
-bert/encoder/layer_1/intermediate/dense/mul_2Mul/bert/encoder/layer_1/intermediate/dense/mul_2/x-bert/encoder/layer_1/intermediate/dense/add_1*
T0* 
_output_shapes
:

¿
-bert/encoder/layer_1/intermediate/dense/mul_3Mul/bert/encoder/layer_1/intermediate/dense/BiasAdd-bert/encoder/layer_1/intermediate/dense/mul_2* 
_output_shapes
:
*
T0
Ù
Kbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel
Î
Lbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
seed2 
Ý
Ibert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:

Ý
(bert/encoder/layer_1/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
	container *
shape:

»
/bert/encoder/layer_1/output/dense/kernel/AssignAssign(bert/encoder/layer_1/output/dense/kernelEbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
validate_shape(* 
_output_shapes
:

Ë
-bert/encoder/layer_1/output/dense/kernel/readIdentity(bert/encoder/layer_1/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:

Â
8bert/encoder/layer_1/output/dense/bias/Initializer/zerosConst*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
dtype0*
_output_shapes	
:
Ï
&bert/encoder/layer_1/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
	container *
shape:
£
-bert/encoder/layer_1/output/dense/bias/AssignAssign&bert/encoder/layer_1/output/dense/bias8bert/encoder/layer_1/output/dense/bias/Initializer/zeros*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
À
+bert/encoder/layer_1/output/dense/bias/readIdentity&bert/encoder/layer_1/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
_output_shapes	
:
á
(bert/encoder/layer_1/output/dense/MatMulMatMul-bert/encoder/layer_1/intermediate/dense/mul_3-bert/encoder/layer_1/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
Í
)bert/encoder/layer_1/output/dense/BiasAddBiasAdd(bert/encoder/layer_1/output/dense/MatMul+bert/encoder/layer_1/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

½
bert/encoder/layer_1/output/addAdd)bert/encoder/layer_1/output/dense/BiasAdd?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ê
<bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_1/output/LayerNorm/beta
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta
³
1bert/encoder/layer_1/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_1/output/LayerNorm/beta<bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta
Ì
/bert/encoder/layer_1/output/LayerNorm/beta/readIdentity*bert/encoder/layer_1/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
_output_shapes	
:
Ë
<bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
dtype0
Ù
+bert/encoder/layer_1/output/LayerNorm/gamma
VariableV2*
shared_name *>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
¶
2bert/encoder/layer_1/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_1/output/LayerNorm/gamma<bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_1/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_1/output/LayerNorm/gamma*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
_output_shapes	
:*
T0

Dbert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
è
2bert/encoder/layer_1/output/LayerNorm/moments/meanMeanbert/encoder/layer_1/output/addDbert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
¨
:bert/encoder/layer_1/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_1/output/LayerNorm/moments/mean*
_output_shapes
:	*
T0
Ü
?bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_1/output/add:bert/encoder/layer_1/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Hbert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

6bert/encoder/layer_1/output/LayerNorm/moments/varianceMean?bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
z
5bert/encoder/layer_1/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_1/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_1/output/LayerNorm/moments/variance5bert/encoder/layer_1/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0

5bert/encoder/layer_1/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_1/output/LayerNorm/batchnorm/add*
_output_shapes
:	*
T0
Î
3bert/encoder/layer_1/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_1/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

½
5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_1/output/add3bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_1/output/LayerNorm/moments/mean3bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Í
3bert/encoder/layer_1/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_1/output/LayerNorm/beta/read5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ó
5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_1/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
é
Sbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/shape*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:

ý
Qbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel* 
_output_shapes
:

í
0bert/encoder/layer_2/attention/self/query/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
	container *
shape:

Û
7bert/encoder/layer_2/attention/self/query/kernel/AssignAssign0bert/encoder/layer_2/attention/self/query/kernelMbert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
validate_shape(
ã
5bert/encoder/layer_2/attention/self/query/kernel/readIdentity0bert/encoder/layer_2/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_2/attention/self/query/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
dtype0
ß
.bert/encoder/layer_2/attention/self/query/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ã
5bert/encoder/layer_2/attention/self/query/bias/AssignAssign.bert/encoder/layer_2/attention/self/query/bias@bert/encoder/layer_2/attention/self/query/bias/Initializer/zeros*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ø
3bert/encoder/layer_2/attention/self/query/bias/readIdentity.bert/encoder/layer_2/attention/self/query/bias*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
_output_shapes	
:*
T0
ù
0bert/encoder/layer_2/attention/self/query/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_15bert/encoder/layer_2/attention/self/query/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
å
1bert/encoder/layer_2/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_2/attention/self/query/MatMul3bert/encoder/layer_2/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

å
Qbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
dtype0
Ø
Pbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/shape*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0
õ
Obert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel
ã
Kbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel
é
.bert/encoder/layer_2/attention/self/key/kernel
VariableV2*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ó
5bert/encoder/layer_2/attention/self/key/kernel/AssignAssign.bert/encoder/layer_2/attention/self/key/kernelKbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_2/attention/self/key/kernel/readIdentity.bert/encoder/layer_2/attention/self/key/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel* 
_output_shapes
:

Î
>bert/encoder/layer_2/attention/self/key/bias/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
dtype0*
_output_shapes	
:
Û
,bert/encoder/layer_2/attention/self/key/bias
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
»
3bert/encoder/layer_2/attention/self/key/bias/AssignAssign,bert/encoder/layer_2/attention/self/key/bias>bert/encoder/layer_2/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_2/attention/self/key/bias/readIdentity,bert/encoder/layer_2/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
_output_shapes	
:
õ
.bert/encoder/layer_2/attention/self/key/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_13bert/encoder/layer_2/attention/self/key/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
ß
/bert/encoder/layer_2/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_2/attention/self/key/MatMul1bert/encoder/layer_2/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

é
Sbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ý
Qbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel
ë
Mbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel* 
_output_shapes
:

í
0bert/encoder/layer_2/attention/self/value/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
	container 
Û
7bert/encoder/layer_2/attention/self/value/kernel/AssignAssign0bert/encoder/layer_2/attention/self/value/kernelMbert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_2/attention/self/value/kernel/readIdentity0bert/encoder/layer_2/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_2/attention/self/value/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_2/attention/self/value/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
	container *
shape:
Ã
5bert/encoder/layer_2/attention/self/value/bias/AssignAssign.bert/encoder/layer_2/attention/self/value/bias@bert/encoder/layer_2/attention/self/value/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
validate_shape(
Ø
3bert/encoder/layer_2/attention/self/value/bias/readIdentity.bert/encoder/layer_2/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
_output_shapes	
:
ù
0bert/encoder/layer_2/attention/self/value/MatMulMatMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_15bert/encoder/layer_2/attention/self/value/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
å
1bert/encoder/layer_2/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_2/attention/self/value/MatMul3bert/encoder/layer_2/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:


1bert/encoder/layer_2/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Ü
+bert/encoder/layer_2/attention/self/ReshapeReshape1bert/encoder/layer_2/attention/self/query/BiasAdd1bert/encoder/layer_2/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:@

2bert/encoder/layer_2/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_2/attention/self/transpose	Transpose+bert/encoder/layer_2/attention/self/Reshape2bert/encoder/layer_2/attention/self/transpose/perm*
Tperm0*
T0*'
_output_shapes
:@

3bert/encoder/layer_2/attention/self/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Þ
-bert/encoder/layer_2/attention/self/Reshape_1Reshape/bert/encoder/layer_2/attention/self/key/BiasAdd3bert/encoder/layer_2/attention/self/Reshape_1/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_2/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_2/attention/self/transpose_1	Transpose-bert/encoder/layer_2/attention/self/Reshape_14bert/encoder/layer_2/attention/self/transpose_1/perm*
T0*'
_output_shapes
:@*
Tperm0
è
*bert/encoder/layer_2/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_2/attention/self/transpose/bert/encoder/layer_2/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:
n
)bert/encoder/layer_2/attention/self/Mul/yConst*
dtype0*
_output_shapes
: *
valueB
 *   >
¸
'bert/encoder/layer_2/attention/self/MulMul*bert/encoder/layer_2/attention/self/MatMul)bert/encoder/layer_2/attention/self/Mul/y*(
_output_shapes
:*
T0
|
2bert/encoder/layer_2/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Á
.bert/encoder/layer_2/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_2/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:
n
)bert/encoder/layer_2/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¼
'bert/encoder/layer_2/attention/self/subSub)bert/encoder/layer_2/attention/self/sub/x.bert/encoder/layer_2/attention/self/ExpandDims*(
_output_shapes
:*
T0
p
+bert/encoder/layer_2/attention/self/mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 * @Æ
¹
)bert/encoder/layer_2/attention/self/mul_1Mul'bert/encoder/layer_2/attention/self/sub+bert/encoder/layer_2/attention/self/mul_1/y*
T0*(
_output_shapes
:
µ
'bert/encoder/layer_2/attention/self/addAdd'bert/encoder/layer_2/attention/self/Mul)bert/encoder/layer_2/attention/self/mul_1*(
_output_shapes
:*
T0

+bert/encoder/layer_2/attention/self/SoftmaxSoftmax'bert/encoder/layer_2/attention/self/add*
T0*(
_output_shapes
:

3bert/encoder/layer_2/attention/self/Reshape_2/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
à
-bert/encoder/layer_2/attention/self/Reshape_2Reshape1bert/encoder/layer_2/attention/self/value/BiasAdd3bert/encoder/layer_2/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_2/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_2/attention/self/transpose_2	Transpose-bert/encoder/layer_2/attention/self/Reshape_24bert/encoder/layer_2/attention/self/transpose_2/perm*'
_output_shapes
:@*
Tperm0*
T0
ç
,bert/encoder/layer_2/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_2/attention/self/Softmax/bert/encoder/layer_2/attention/self/transpose_2*'
_output_shapes
:@*
adj_x( *
adj_y( *
T0

4bert/encoder/layer_2/attention/self/transpose_3/permConst*
dtype0*
_output_shapes
:*%
valueB"             
ß
/bert/encoder/layer_2/attention/self/transpose_3	Transpose,bert/encoder/layer_2/attention/self/MatMul_14bert/encoder/layer_2/attention/self/transpose_3/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_2/attention/self/Reshape_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
×
-bert/encoder/layer_2/attention/self/Reshape_3Reshape/bert/encoder/layer_2/attention/self/transpose_33bert/encoder/layer_2/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:

í
Ubert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel
à
Tbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel
â
Vbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
dtype0
á
_bert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
seed2 

Sbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel
ñ
2bert/encoder/layer_2/attention/output/dense/kernel
VariableV2*
shared_name *E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ã
9bert/encoder/layer_2/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_2/attention/output/dense/kernelObert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
é
7bert/encoder/layer_2/attention/output/dense/kernel/readIdentity2bert/encoder/layer_2/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel* 
_output_shapes
:

Ö
Bbert/encoder/layer_2/attention/output/dense/bias/Initializer/zerosConst*
valueB*    *C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
dtype0*
_output_shapes	
:
ã
0bert/encoder/layer_2/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
	container *
shape:
Ë
7bert/encoder/layer_2/attention/output/dense/bias/AssignAssign0bert/encoder/layer_2/attention/output/dense/biasBbert/encoder/layer_2/attention/output/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias
Þ
5bert/encoder/layer_2/attention/output/dense/bias/readIdentity0bert/encoder/layer_2/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
_output_shapes	
:
õ
2bert/encoder/layer_2/attention/output/dense/MatMulMatMul-bert/encoder/layer_2/attention/self/Reshape_37bert/encoder/layer_2/attention/output/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
ë
3bert/encoder/layer_2/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_2/attention/output/dense/MatMul5bert/encoder/layer_2/attention/output/dense/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC
Ç
)bert/encoder/layer_2/attention/output/addAdd3bert/encoder/layer_2/attention/output/dense/BiasAdd5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Þ
Fbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
ë
4bert/encoder/layer_2/attention/output/LayerNorm/beta
VariableV2*
shared_name *G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
Û
;bert/encoder/layer_2/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_2/attention/output/LayerNorm/betaFbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ê
9bert/encoder/layer_2/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_2/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
_output_shapes	
:
ß
Fbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
dtype0
í
5bert/encoder/layer_2/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
	container *
shape:
Þ
<bert/encoder/layer_2/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_2/attention/output/LayerNorm/gammaFbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma
í
:bert/encoder/layer_2/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_2/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
_output_shapes	
:

Nbert/encoder/layer_2/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

<bert/encoder/layer_2/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_2/attention/output/addNbert/encoder/layer_2/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
¼
Dbert/encoder/layer_2/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_2/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ú
Ibert/encoder/layer_2/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_2/attention/output/addDbert/encoder/layer_2/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Rbert/encoder/layer_2/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_2/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_2/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_2/attention/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	

?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_2/attention/output/LayerNorm/moments/variance?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0
±
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add*
_output_shapes
:	*
T0
ì
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_2/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Û
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_2/attention/output/add=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_2/attention/output/LayerNorm/moments/mean=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

ë
=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_2/attention/output/LayerNorm/beta/read?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
*
T0
ñ
?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

å
Qbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
õ
Obert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel* 
_output_shapes
:

é
.bert/encoder/layer_2/intermediate/dense/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
	container 
Ó
5bert/encoder/layer_2/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_2/intermediate/dense/kernelKbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel
Ý
3bert/encoder/layer_2/intermediate/dense/kernel/readIdentity.bert/encoder/layer_2/intermediate/dense/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel
Ú
Nbert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ê
Dbert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias
Õ
>bert/encoder/layer_2/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias
Û
,bert/encoder/layer_2/intermediate/dense/bias
VariableV2*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
»
3bert/encoder/layer_2/intermediate/dense/bias/AssignAssign,bert/encoder/layer_2/intermediate/dense/bias>bert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
validate_shape(
Ò
1bert/encoder/layer_2/intermediate/dense/bias/readIdentity,bert/encoder/layer_2/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
_output_shapes	
:
ÿ
.bert/encoder/layer_2/intermediate/dense/MatMulMatMul?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_2/intermediate/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
ß
/bert/encoder/layer_2/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_2/intermediate/dense/MatMul1bert/encoder/layer_2/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_2/intermediate/dense/Pow/yConst*
_output_shapes
: *
valueB
 *  @@*
dtype0
½
+bert/encoder/layer_2/intermediate/dense/PowPow/bert/encoder/layer_2/intermediate/dense/BiasAdd-bert/encoder/layer_2/intermediate/dense/Pow/y*
T0* 
_output_shapes
:

r
-bert/encoder/layer_2/intermediate/dense/mul/xConst*
_output_shapes
: *
valueB
 *'7=*
dtype0
¹
+bert/encoder/layer_2/intermediate/dense/mulMul-bert/encoder/layer_2/intermediate/dense/mul/x+bert/encoder/layer_2/intermediate/dense/Pow*
T0* 
_output_shapes
:

»
+bert/encoder/layer_2/intermediate/dense/addAdd/bert/encoder/layer_2/intermediate/dense/BiasAdd+bert/encoder/layer_2/intermediate/dense/mul* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_2/intermediate/dense/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 **BL?
½
-bert/encoder/layer_2/intermediate/dense/mul_1Mul/bert/encoder/layer_2/intermediate/dense/mul_1/x+bert/encoder/layer_2/intermediate/dense/add* 
_output_shapes
:
*
T0

,bert/encoder/layer_2/intermediate/dense/TanhTanh-bert/encoder/layer_2/intermediate/dense/mul_1*
T0* 
_output_shapes
:

t
/bert/encoder/layer_2/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¾
-bert/encoder/layer_2/intermediate/dense/add_1Add/bert/encoder/layer_2/intermediate/dense/add_1/x,bert/encoder/layer_2/intermediate/dense/Tanh*
T0* 
_output_shapes
:

t
/bert/encoder/layer_2/intermediate/dense/mul_2/xConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
¿
-bert/encoder/layer_2/intermediate/dense/mul_2Mul/bert/encoder/layer_2/intermediate/dense/mul_2/x-bert/encoder/layer_2/intermediate/dense/add_1* 
_output_shapes
:
*
T0
¿
-bert/encoder/layer_2/intermediate/dense/mul_3Mul/bert/encoder/layer_2/intermediate/dense/BiasAdd-bert/encoder/layer_2/intermediate/dense/mul_2*
T0* 
_output_shapes
:

Ù
Kbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
dtype0
Ì
Jbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
dtype0*
_output_shapes
: 
Î
Lbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/shape*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ý
Ibert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/stddev*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel* 
_output_shapes
:
*
T0
Ë
Ebert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel* 
_output_shapes
:

Ý
(bert/encoder/layer_2/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
	container *
shape:

»
/bert/encoder/layer_2/output/dense/kernel/AssignAssign(bert/encoder/layer_2/output/dense/kernelEbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ë
-bert/encoder/layer_2/output/dense/kernel/readIdentity(bert/encoder/layer_2/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel* 
_output_shapes
:

Â
8bert/encoder/layer_2/output/dense/bias/Initializer/zerosConst*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
dtype0*
_output_shapes	
:
Ï
&bert/encoder/layer_2/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
	container *
shape:
£
-bert/encoder/layer_2/output/dense/bias/AssignAssign&bert/encoder/layer_2/output/dense/bias8bert/encoder/layer_2/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
validate_shape(*
_output_shapes	
:
À
+bert/encoder/layer_2/output/dense/bias/readIdentity&bert/encoder/layer_2/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
_output_shapes	
:
á
(bert/encoder/layer_2/output/dense/MatMulMatMul-bert/encoder/layer_2/intermediate/dense/mul_3-bert/encoder/layer_2/output/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
Í
)bert/encoder/layer_2/output/dense/BiasAddBiasAdd(bert/encoder/layer_2/output/dense/MatMul+bert/encoder/layer_2/output/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
½
bert/encoder/layer_2/output/addAdd)bert/encoder/layer_2/output/dense/BiasAdd?bert/encoder/layer_2/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ê
<bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_2/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
	container *
shape:
³
1bert/encoder/layer_2/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_2/output/LayerNorm/beta<bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zeros*
T0*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
Ì
/bert/encoder/layer_2/output/LayerNorm/beta/readIdentity*bert/encoder/layer_2/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
_output_shapes	
:
Ë
<bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma
Ù
+bert/encoder/layer_2/output/LayerNorm/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
	container 
¶
2bert/encoder/layer_2/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_2/output/LayerNorm/gamma<bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_2/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_2/output/LayerNorm/gamma*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
_output_shapes	
:*
T0

Dbert/encoder/layer_2/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
è
2bert/encoder/layer_2/output/LayerNorm/moments/meanMeanbert/encoder/layer_2/output/addDbert/encoder/layer_2/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
¨
:bert/encoder/layer_2/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_2/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_2/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_2/output/add:bert/encoder/layer_2/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Hbert/encoder/layer_2/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

6bert/encoder/layer_2/output/LayerNorm/moments/varianceMean?bert/encoder/layer_2/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_2/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
z
5bert/encoder/layer_2/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_2/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_2/output/LayerNorm/moments/variance5bert/encoder/layer_2/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0

5bert/encoder/layer_2/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_2/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
Î
3bert/encoder/layer_2/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_2/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

½
5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_2/output/add3bert/encoder/layer_2/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_2/output/LayerNorm/moments/mean3bert/encoder/layer_2/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Í
3bert/encoder/layer_2/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_2/output/LayerNorm/beta/read5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ó
5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_2/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_2/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
é
Sbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
dtype0
Ü
Rbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
seed2 
ý
Qbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel* 
_output_shapes
:

í
0bert/encoder/layer_3/attention/self/query/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
	container *
shape:

Û
7bert/encoder/layer_3/attention/self/query/kernel/AssignAssign0bert/encoder/layer_3/attention/self/query/kernelMbert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ã
5bert/encoder/layer_3/attention/self/query/kernel/readIdentity0bert/encoder/layer_3/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_3/attention/self/query/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_3/attention/self/query/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ã
5bert/encoder/layer_3/attention/self/query/bias/AssignAssign.bert/encoder/layer_3/attention/self/query/bias@bert/encoder/layer_3/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_3/attention/self/query/bias/readIdentity.bert/encoder/layer_3/attention/self/query/bias*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
_output_shapes	
:*
T0
ù
0bert/encoder/layer_3/attention/self/query/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_15bert/encoder/layer_3/attention/self/query/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
å
1bert/encoder/layer_3/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_3/attention/self/query/MatMul3bert/encoder/layer_3/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

å
Qbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel
Ø
Pbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
dtype0
Õ
[bert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
seed2 
õ
Obert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel
ã
Kbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel* 
_output_shapes
:

é
.bert/encoder/layer_3/attention/self/key/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
	container 
Ó
5bert/encoder/layer_3/attention/self/key/kernel/AssignAssign.bert/encoder/layer_3/attention/self/key/kernelKbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_3/attention/self/key/kernel/readIdentity.bert/encoder/layer_3/attention/self/key/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel* 
_output_shapes
:

Î
>bert/encoder/layer_3/attention/self/key/bias/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
dtype0*
_output_shapes	
:
Û
,bert/encoder/layer_3/attention/self/key/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias
»
3bert/encoder/layer_3/attention/self/key/bias/AssignAssign,bert/encoder/layer_3/attention/self/key/bias>bert/encoder/layer_3/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_3/attention/self/key/bias/readIdentity,bert/encoder/layer_3/attention/self/key/bias*
_output_shapes	
:*
T0*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias
õ
.bert/encoder/layer_3/attention/self/key/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_13bert/encoder/layer_3/attention/self/key/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ß
/bert/encoder/layer_3/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_3/attention/self/key/MatMul1bert/encoder/layer_3/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

é
Sbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
dtype0
Û
]bert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
seed2 
ý
Qbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel* 
_output_shapes
:

í
0bert/encoder/layer_3/attention/self/value/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Û
7bert/encoder/layer_3/attention/self/value/kernel/AssignAssign0bert/encoder/layer_3/attention/self/value/kernelMbert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_3/attention/self/value/kernel/readIdentity0bert/encoder/layer_3/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_3/attention/self/value/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_3/attention/self/value/bias
VariableV2*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ã
5bert/encoder/layer_3/attention/self/value/bias/AssignAssign.bert/encoder/layer_3/attention/self/value/bias@bert/encoder/layer_3/attention/self/value/bias/Initializer/zeros*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ø
3bert/encoder/layer_3/attention/self/value/bias/readIdentity.bert/encoder/layer_3/attention/self/value/bias*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
_output_shapes	
:*
T0
ù
0bert/encoder/layer_3/attention/self/value/MatMulMatMul5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_15bert/encoder/layer_3/attention/self/value/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
å
1bert/encoder/layer_3/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_3/attention/self/value/MatMul3bert/encoder/layer_3/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0

1bert/encoder/layer_3/attention/self/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
Ü
+bert/encoder/layer_3/attention/self/ReshapeReshape1bert/encoder/layer_3/attention/self/query/BiasAdd1bert/encoder/layer_3/attention/self/Reshape/shape*'
_output_shapes
:@*
T0*
Tshape0

2bert/encoder/layer_3/attention/self/transpose/permConst*
_output_shapes
:*%
valueB"             *
dtype0
Ú
-bert/encoder/layer_3/attention/self/transpose	Transpose+bert/encoder/layer_3/attention/self/Reshape2bert/encoder/layer_3/attention/self/transpose/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_3/attention/self/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Þ
-bert/encoder/layer_3/attention/self/Reshape_1Reshape/bert/encoder/layer_3/attention/self/key/BiasAdd3bert/encoder/layer_3/attention/self/Reshape_1/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_3/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_3/attention/self/transpose_1	Transpose-bert/encoder/layer_3/attention/self/Reshape_14bert/encoder/layer_3/attention/self/transpose_1/perm*
T0*'
_output_shapes
:@*
Tperm0
è
*bert/encoder/layer_3/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_3/attention/self/transpose/bert/encoder/layer_3/attention/self/transpose_1*
T0*(
_output_shapes
:*
adj_x( *
adj_y(
n
)bert/encoder/layer_3/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
¸
'bert/encoder/layer_3/attention/self/MulMul*bert/encoder/layer_3/attention/self/MatMul)bert/encoder/layer_3/attention/self/Mul/y*
T0*(
_output_shapes
:
|
2bert/encoder/layer_3/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Á
.bert/encoder/layer_3/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_3/attention/self/ExpandDims/dim*
T0*(
_output_shapes
:*

Tdim0
n
)bert/encoder/layer_3/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¼
'bert/encoder/layer_3/attention/self/subSub)bert/encoder/layer_3/attention/self/sub/x.bert/encoder/layer_3/attention/self/ExpandDims*(
_output_shapes
:*
T0
p
+bert/encoder/layer_3/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¹
)bert/encoder/layer_3/attention/self/mul_1Mul'bert/encoder/layer_3/attention/self/sub+bert/encoder/layer_3/attention/self/mul_1/y*
T0*(
_output_shapes
:
µ
'bert/encoder/layer_3/attention/self/addAdd'bert/encoder/layer_3/attention/self/Mul)bert/encoder/layer_3/attention/self/mul_1*(
_output_shapes
:*
T0

+bert/encoder/layer_3/attention/self/SoftmaxSoftmax'bert/encoder/layer_3/attention/self/add*
T0*(
_output_shapes
:

3bert/encoder/layer_3/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
à
-bert/encoder/layer_3/attention/self/Reshape_2Reshape1bert/encoder/layer_3/attention/self/value/BiasAdd3bert/encoder/layer_3/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_3/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_3/attention/self/transpose_2	Transpose-bert/encoder/layer_3/attention/self/Reshape_24bert/encoder/layer_3/attention/self/transpose_2/perm*
T0*'
_output_shapes
:@*
Tperm0
ç
,bert/encoder/layer_3/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_3/attention/self/Softmax/bert/encoder/layer_3/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:@

4bert/encoder/layer_3/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ß
/bert/encoder/layer_3/attention/self/transpose_3	Transpose,bert/encoder/layer_3/attention/self/MatMul_14bert/encoder/layer_3/attention/self/transpose_3/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_3/attention/self/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
×
-bert/encoder/layer_3/attention/self/Reshape_3Reshape/bert/encoder/layer_3/attention/self/transpose_33bert/encoder/layer_3/attention/self/Reshape_3/shape*
Tshape0* 
_output_shapes
:
*
T0
í
Ubert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
dtype0
à
Tbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
dtype0
â
Vbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
á
_bert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
seed2 

Sbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel
ó
Obert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel* 
_output_shapes
:

ñ
2bert/encoder/layer_3/attention/output/dense/kernel
VariableV2*
shared_name *E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ã
9bert/encoder/layer_3/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_3/attention/output/dense/kernelObert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

é
7bert/encoder/layer_3/attention/output/dense/kernel/readIdentity2bert/encoder/layer_3/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel* 
_output_shapes
:

Ö
Bbert/encoder/layer_3/attention/output/dense/bias/Initializer/zerosConst*
valueB*    *C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
dtype0*
_output_shapes	
:
ã
0bert/encoder/layer_3/attention/output/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
	container 
Ë
7bert/encoder/layer_3/attention/output/dense/bias/AssignAssign0bert/encoder/layer_3/attention/output/dense/biasBbert/encoder/layer_3/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
Þ
5bert/encoder/layer_3/attention/output/dense/bias/readIdentity0bert/encoder/layer_3/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
_output_shapes	
:
õ
2bert/encoder/layer_3/attention/output/dense/MatMulMatMul-bert/encoder/layer_3/attention/self/Reshape_37bert/encoder/layer_3/attention/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ë
3bert/encoder/layer_3/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_3/attention/output/dense/MatMul5bert/encoder/layer_3/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

Ç
)bert/encoder/layer_3/attention/output/addAdd3bert/encoder/layer_3/attention/output/dense/BiasAdd5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
*
T0
Þ
Fbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
dtype0
ë
4bert/encoder/layer_3/attention/output/LayerNorm/beta
VariableV2*
shared_name *G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
Û
;bert/encoder/layer_3/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_3/attention/output/LayerNorm/betaFbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ê
9bert/encoder/layer_3/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_3/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
_output_shapes	
:
ß
Fbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_3/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
	container *
shape:
Þ
<bert/encoder/layer_3/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_3/attention/output/LayerNorm/gammaFbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/ones*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
í
:bert/encoder/layer_3/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_3/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
_output_shapes	
:

Nbert/encoder/layer_3/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

<bert/encoder/layer_3/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_3/attention/output/addNbert/encoder/layer_3/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
¼
Dbert/encoder/layer_3/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_3/attention/output/LayerNorm/moments/mean*
_output_shapes
:	*
T0
ú
Ibert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_3/attention/output/addDbert/encoder/layer_3/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Rbert/encoder/layer_3/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_3/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_3/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_3/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0

?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_3/attention/output/LayerNorm/moments/variance?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0
±
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ì
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_3/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Û
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_3/attention/output/add=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_3/attention/output/LayerNorm/moments/mean=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

ë
=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_3/attention/output/LayerNorm/beta/read?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ñ
?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

å
Qbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
seed2 
õ
Obert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel
ã
Kbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel* 
_output_shapes
:

é
.bert/encoder/layer_3/intermediate/dense/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ó
5bert/encoder/layer_3/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_3/intermediate/dense/kernelKbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ý
3bert/encoder/layer_3/intermediate/dense/kernel/readIdentity.bert/encoder/layer_3/intermediate/dense/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel
Ú
Nbert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ê
Dbert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
dtype0*
_output_shapes
: 
Õ
>bert/encoder/layer_3/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros/Const*
T0*

index_type0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
_output_shapes	
:
Û
,bert/encoder/layer_3/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
	container *
shape:
»
3bert/encoder/layer_3/intermediate/dense/bias/AssignAssign,bert/encoder/layer_3/intermediate/dense/bias>bert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
validate_shape(
Ò
1bert/encoder/layer_3/intermediate/dense/bias/readIdentity,bert/encoder/layer_3/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
_output_shapes	
:
ÿ
.bert/encoder/layer_3/intermediate/dense/MatMulMatMul?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_3/intermediate/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ß
/bert/encoder/layer_3/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_3/intermediate/dense/MatMul1bert/encoder/layer_3/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_3/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
½
+bert/encoder/layer_3/intermediate/dense/PowPow/bert/encoder/layer_3/intermediate/dense/BiasAdd-bert/encoder/layer_3/intermediate/dense/Pow/y*
T0* 
_output_shapes
:

r
-bert/encoder/layer_3/intermediate/dense/mul/xConst*
_output_shapes
: *
valueB
 *'7=*
dtype0
¹
+bert/encoder/layer_3/intermediate/dense/mulMul-bert/encoder/layer_3/intermediate/dense/mul/x+bert/encoder/layer_3/intermediate/dense/Pow* 
_output_shapes
:
*
T0
»
+bert/encoder/layer_3/intermediate/dense/addAdd/bert/encoder/layer_3/intermediate/dense/BiasAdd+bert/encoder/layer_3/intermediate/dense/mul* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_3/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
½
-bert/encoder/layer_3/intermediate/dense/mul_1Mul/bert/encoder/layer_3/intermediate/dense/mul_1/x+bert/encoder/layer_3/intermediate/dense/add*
T0* 
_output_shapes
:


,bert/encoder/layer_3/intermediate/dense/TanhTanh-bert/encoder/layer_3/intermediate/dense/mul_1*
T0* 
_output_shapes
:

t
/bert/encoder/layer_3/intermediate/dense/add_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
¾
-bert/encoder/layer_3/intermediate/dense/add_1Add/bert/encoder/layer_3/intermediate/dense/add_1/x,bert/encoder/layer_3/intermediate/dense/Tanh*
T0* 
_output_shapes
:

t
/bert/encoder/layer_3/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
¿
-bert/encoder/layer_3/intermediate/dense/mul_2Mul/bert/encoder/layer_3/intermediate/dense/mul_2/x-bert/encoder/layer_3/intermediate/dense/add_1*
T0* 
_output_shapes
:

¿
-bert/encoder/layer_3/intermediate/dense/mul_3Mul/bert/encoder/layer_3/intermediate/dense/BiasAdd-bert/encoder/layer_3/intermediate/dense/mul_2*
T0* 
_output_shapes
:

Ù
Kbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
dtype0*
_output_shapes
: 
Î
Lbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
seed2 
Ý
Ibert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel
Ý
(bert/encoder/layer_3/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
	container *
shape:

»
/bert/encoder/layer_3/output/dense/kernel/AssignAssign(bert/encoder/layer_3/output/dense/kernelEbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel
Ë
-bert/encoder/layer_3/output/dense/kernel/readIdentity(bert/encoder/layer_3/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel* 
_output_shapes
:

Â
8bert/encoder/layer_3/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias
Ï
&bert/encoder/layer_3/output/dense/bias
VariableV2*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
£
-bert/encoder/layer_3/output/dense/bias/AssignAssign&bert/encoder/layer_3/output/dense/bias8bert/encoder/layer_3/output/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias
À
+bert/encoder/layer_3/output/dense/bias/readIdentity&bert/encoder/layer_3/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
_output_shapes	
:
á
(bert/encoder/layer_3/output/dense/MatMulMatMul-bert/encoder/layer_3/intermediate/dense/mul_3-bert/encoder/layer_3/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
Í
)bert/encoder/layer_3/output/dense/BiasAddBiasAdd(bert/encoder/layer_3/output/dense/MatMul+bert/encoder/layer_3/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

½
bert/encoder/layer_3/output/addAdd)bert/encoder/layer_3/output/dense/BiasAdd?bert/encoder/layer_3/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ê
<bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_3/output/LayerNorm/beta
VariableV2*
shared_name *=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
³
1bert/encoder/layer_3/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_3/output/LayerNorm/beta<bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
validate_shape(
Ì
/bert/encoder/layer_3/output/LayerNorm/beta/readIdentity*bert/encoder/layer_3/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
_output_shapes	
:
Ë
<bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma
Ù
+bert/encoder/layer_3/output/LayerNorm/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
	container 
¶
2bert/encoder/layer_3/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_3/output/LayerNorm/gamma<bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/ones*
T0*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ï
0bert/encoder/layer_3/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_3/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
_output_shapes	
:

Dbert/encoder/layer_3/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
è
2bert/encoder/layer_3/output/LayerNorm/moments/meanMeanbert/encoder/layer_3/output/addDbert/encoder/layer_3/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
¨
:bert/encoder/layer_3/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_3/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_3/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_3/output/add:bert/encoder/layer_3/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Hbert/encoder/layer_3/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

6bert/encoder/layer_3/output/LayerNorm/moments/varianceMean?bert/encoder/layer_3/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_3/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
z
5bert/encoder/layer_3/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_3/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_3/output/LayerNorm/moments/variance5bert/encoder/layer_3/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0

5bert/encoder/layer_3/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_3/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
Î
3bert/encoder/layer_3/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_3/output/LayerNorm/gamma/read* 
_output_shapes
:
*
T0
½
5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_3/output/add3bert/encoder/layer_3/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
Ð
5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_3/output/LayerNorm/moments/mean3bert/encoder/layer_3/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Í
3bert/encoder/layer_3/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_3/output/LayerNorm/beta/read5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ó
5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_3/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_3/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

é
Sbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel
Þ
Tbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ý
Qbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel
í
0bert/encoder/layer_4/attention/self/query/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Û
7bert/encoder/layer_4/attention/self/query/kernel/AssignAssign0bert/encoder/layer_4/attention/self/query/kernelMbert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ã
5bert/encoder/layer_4/attention/self/query/kernel/readIdentity0bert/encoder/layer_4/attention/self/query/kernel*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel* 
_output_shapes
:
*
T0
Ò
@bert/encoder/layer_4/attention/self/query/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_4/attention/self/query/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
	container *
shape:
Ã
5bert/encoder/layer_4/attention/self/query/bias/AssignAssign.bert/encoder/layer_4/attention/self/query/bias@bert/encoder/layer_4/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_4/attention/self/query/bias/readIdentity.bert/encoder/layer_4/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
_output_shapes	
:
ù
0bert/encoder/layer_4/attention/self/query/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_15bert/encoder/layer_4/attention/self/query/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
å
1bert/encoder/layer_4/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_4/attention/self/query/MatMul3bert/encoder/layer_4/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

å
Qbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
dtype0
Ø
Pbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel
Õ
[bert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
õ
Obert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel
ã
Kbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel
é
.bert/encoder/layer_4/attention/self/key/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ó
5bert/encoder/layer_4/attention/self/key/kernel/AssignAssign.bert/encoder/layer_4/attention/self/key/kernelKbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_4/attention/self/key/kernel/readIdentity.bert/encoder/layer_4/attention/self/key/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel* 
_output_shapes
:

Î
>bert/encoder/layer_4/attention/self/key/bias/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
dtype0*
_output_shapes	
:
Û
,bert/encoder/layer_4/attention/self/key/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
	container 
»
3bert/encoder/layer_4/attention/self/key/bias/AssignAssign,bert/encoder/layer_4/attention/self/key/bias>bert/encoder/layer_4/attention/self/key/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias
Ò
1bert/encoder/layer_4/attention/self/key/bias/readIdentity,bert/encoder/layer_4/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
_output_shapes	
:
õ
.bert/encoder/layer_4/attention/self/key/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_13bert/encoder/layer_4/attention/self/key/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ß
/bert/encoder/layer_4/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_4/attention/self/key/MatMul1bert/encoder/layer_4/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

é
Sbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
dtype0
Ü
Rbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel
ý
Qbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel* 
_output_shapes
:

í
0bert/encoder/layer_4/attention/self/value/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Û
7bert/encoder/layer_4/attention/self/value/kernel/AssignAssign0bert/encoder/layer_4/attention/self/value/kernelMbert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ã
5bert/encoder/layer_4/attention/self/value/kernel/readIdentity0bert/encoder/layer_4/attention/self/value/kernel*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel* 
_output_shapes
:
*
T0
Ò
@bert/encoder/layer_4/attention/self/value/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_4/attention/self/value/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias
Ã
5bert/encoder/layer_4/attention/self/value/bias/AssignAssign.bert/encoder/layer_4/attention/self/value/bias@bert/encoder/layer_4/attention/self/value/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_4/attention/self/value/bias/readIdentity.bert/encoder/layer_4/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
_output_shapes	
:
ù
0bert/encoder/layer_4/attention/self/value/MatMulMatMul5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_15bert/encoder/layer_4/attention/self/value/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_4/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_4/attention/self/value/MatMul3bert/encoder/layer_4/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0

1bert/encoder/layer_4/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Ü
+bert/encoder/layer_4/attention/self/ReshapeReshape1bert/encoder/layer_4/attention/self/query/BiasAdd1bert/encoder/layer_4/attention/self/Reshape/shape*'
_output_shapes
:@*
T0*
Tshape0

2bert/encoder/layer_4/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_4/attention/self/transpose	Transpose+bert/encoder/layer_4/attention/self/Reshape2bert/encoder/layer_4/attention/self/transpose/perm*
T0*'
_output_shapes
:@*
Tperm0

3bert/encoder/layer_4/attention/self/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Þ
-bert/encoder/layer_4/attention/self/Reshape_1Reshape/bert/encoder/layer_4/attention/self/key/BiasAdd3bert/encoder/layer_4/attention/self/Reshape_1/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_4/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_4/attention/self/transpose_1	Transpose-bert/encoder/layer_4/attention/self/Reshape_14bert/encoder/layer_4/attention/self/transpose_1/perm*'
_output_shapes
:@*
Tperm0*
T0
è
*bert/encoder/layer_4/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_4/attention/self/transpose/bert/encoder/layer_4/attention/self/transpose_1*(
_output_shapes
:*
adj_x( *
adj_y(*
T0
n
)bert/encoder/layer_4/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
¸
'bert/encoder/layer_4/attention/self/MulMul*bert/encoder/layer_4/attention/self/MatMul)bert/encoder/layer_4/attention/self/Mul/y*(
_output_shapes
:*
T0
|
2bert/encoder/layer_4/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Á
.bert/encoder/layer_4/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_4/attention/self/ExpandDims/dim*(
_output_shapes
:*

Tdim0*
T0
n
)bert/encoder/layer_4/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¼
'bert/encoder/layer_4/attention/self/subSub)bert/encoder/layer_4/attention/self/sub/x.bert/encoder/layer_4/attention/self/ExpandDims*
T0*(
_output_shapes
:
p
+bert/encoder/layer_4/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¹
)bert/encoder/layer_4/attention/self/mul_1Mul'bert/encoder/layer_4/attention/self/sub+bert/encoder/layer_4/attention/self/mul_1/y*
T0*(
_output_shapes
:
µ
'bert/encoder/layer_4/attention/self/addAdd'bert/encoder/layer_4/attention/self/Mul)bert/encoder/layer_4/attention/self/mul_1*
T0*(
_output_shapes
:

+bert/encoder/layer_4/attention/self/SoftmaxSoftmax'bert/encoder/layer_4/attention/self/add*
T0*(
_output_shapes
:

3bert/encoder/layer_4/attention/self/Reshape_2/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
à
-bert/encoder/layer_4/attention/self/Reshape_2Reshape1bert/encoder/layer_4/attention/self/value/BiasAdd3bert/encoder/layer_4/attention/self/Reshape_2/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_4/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_4/attention/self/transpose_2	Transpose-bert/encoder/layer_4/attention/self/Reshape_24bert/encoder/layer_4/attention/self/transpose_2/perm*
T0*'
_output_shapes
:@*
Tperm0
ç
,bert/encoder/layer_4/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_4/attention/self/Softmax/bert/encoder/layer_4/attention/self/transpose_2*'
_output_shapes
:@*
adj_x( *
adj_y( *
T0

4bert/encoder/layer_4/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ß
/bert/encoder/layer_4/attention/self/transpose_3	Transpose,bert/encoder/layer_4/attention/self/MatMul_14bert/encoder/layer_4/attention/self/transpose_3/perm*
T0*'
_output_shapes
:@*
Tperm0

3bert/encoder/layer_4/attention/self/Reshape_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
×
-bert/encoder/layer_4/attention/self/Reshape_3Reshape/bert/encoder/layer_4/attention/self/transpose_33bert/encoder/layer_4/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:

í
Ubert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
dtype0
à
Tbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
â
Vbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
á
_bert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:


Sbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel* 
_output_shapes
:

ñ
2bert/encoder/layer_4/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
	container *
shape:

ã
9bert/encoder/layer_4/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_4/attention/output/dense/kernelObert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

é
7bert/encoder/layer_4/attention/output/dense/kernel/readIdentity2bert/encoder/layer_4/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel* 
_output_shapes
:

Ö
Bbert/encoder/layer_4/attention/output/dense/bias/Initializer/zerosConst*
valueB*    *C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
dtype0*
_output_shapes	
:
ã
0bert/encoder/layer_4/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
	container *
shape:
Ë
7bert/encoder/layer_4/attention/output/dense/bias/AssignAssign0bert/encoder/layer_4/attention/output/dense/biasBbert/encoder/layer_4/attention/output/dense/bias/Initializer/zeros*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Þ
5bert/encoder/layer_4/attention/output/dense/bias/readIdentity0bert/encoder/layer_4/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
_output_shapes	
:
õ
2bert/encoder/layer_4/attention/output/dense/MatMulMatMul-bert/encoder/layer_4/attention/self/Reshape_37bert/encoder/layer_4/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ë
3bert/encoder/layer_4/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_4/attention/output/dense/MatMul5bert/encoder/layer_4/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

Ç
)bert/encoder/layer_4/attention/output/addAdd3bert/encoder/layer_4/attention/output/dense/BiasAdd5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Þ
Fbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
ë
4bert/encoder/layer_4/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
	container *
shape:
Û
;bert/encoder/layer_4/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_4/attention/output/LayerNorm/betaFbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta
ê
9bert/encoder/layer_4/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_4/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
_output_shapes	
:
ß
Fbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
dtype0
í
5bert/encoder/layer_4/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
	container *
shape:
Þ
<bert/encoder/layer_4/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_4/attention/output/LayerNorm/gammaFbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
í
:bert/encoder/layer_4/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_4/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
_output_shapes	
:

Nbert/encoder/layer_4/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

<bert/encoder/layer_4/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_4/attention/output/addNbert/encoder/layer_4/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¼
Dbert/encoder/layer_4/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_4/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ú
Ibert/encoder/layer_4/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_4/attention/output/addDbert/encoder/layer_4/attention/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
*
T0

Rbert/encoder/layer_4/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_4/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_4/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_4/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0

?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_4/attention/output/LayerNorm/moments/variance?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	
±
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ì
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_4/attention/output/LayerNorm/gamma/read* 
_output_shapes
:
*
T0
Û
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_4/attention/output/add=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
î
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_4/attention/output/LayerNorm/moments/mean=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

ë
=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_4/attention/output/LayerNorm/beta/read?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ñ
?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

å
Qbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
seed2 
õ
Obert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel* 
_output_shapes
:

é
.bert/encoder/layer_4/intermediate/dense/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
	container 
Ó
5bert/encoder/layer_4/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_4/intermediate/dense/kernelKbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
validate_shape(
Ý
3bert/encoder/layer_4/intermediate/dense/kernel/readIdentity.bert/encoder/layer_4/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel* 
_output_shapes
:

Ú
Nbert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ê
Dbert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias
Õ
>bert/encoder/layer_4/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros/Const*

index_type0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
_output_shapes	
:*
T0
Û
,bert/encoder/layer_4/intermediate/dense/bias
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
»
3bert/encoder/layer_4/intermediate/dense/bias/AssignAssign,bert/encoder/layer_4/intermediate/dense/bias>bert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros*
T0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ò
1bert/encoder/layer_4/intermediate/dense/bias/readIdentity,bert/encoder/layer_4/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
_output_shapes	
:
ÿ
.bert/encoder/layer_4/intermediate/dense/MatMulMatMul?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_4/intermediate/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
ß
/bert/encoder/layer_4/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_4/intermediate/dense/MatMul1bert/encoder/layer_4/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_4/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
½
+bert/encoder/layer_4/intermediate/dense/PowPow/bert/encoder/layer_4/intermediate/dense/BiasAdd-bert/encoder/layer_4/intermediate/dense/Pow/y*
T0* 
_output_shapes
:

r
-bert/encoder/layer_4/intermediate/dense/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *'7=
¹
+bert/encoder/layer_4/intermediate/dense/mulMul-bert/encoder/layer_4/intermediate/dense/mul/x+bert/encoder/layer_4/intermediate/dense/Pow* 
_output_shapes
:
*
T0
»
+bert/encoder/layer_4/intermediate/dense/addAdd/bert/encoder/layer_4/intermediate/dense/BiasAdd+bert/encoder/layer_4/intermediate/dense/mul*
T0* 
_output_shapes
:

t
/bert/encoder/layer_4/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
½
-bert/encoder/layer_4/intermediate/dense/mul_1Mul/bert/encoder/layer_4/intermediate/dense/mul_1/x+bert/encoder/layer_4/intermediate/dense/add*
T0* 
_output_shapes
:


,bert/encoder/layer_4/intermediate/dense/TanhTanh-bert/encoder/layer_4/intermediate/dense/mul_1*
T0* 
_output_shapes
:

t
/bert/encoder/layer_4/intermediate/dense/add_1/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
¾
-bert/encoder/layer_4/intermediate/dense/add_1Add/bert/encoder/layer_4/intermediate/dense/add_1/x,bert/encoder/layer_4/intermediate/dense/Tanh* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_4/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
¿
-bert/encoder/layer_4/intermediate/dense/mul_2Mul/bert/encoder/layer_4/intermediate/dense/mul_2/x-bert/encoder/layer_4/intermediate/dense/add_1*
T0* 
_output_shapes
:

¿
-bert/encoder/layer_4/intermediate/dense/mul_3Mul/bert/encoder/layer_4/intermediate/dense/BiasAdd-bert/encoder/layer_4/intermediate/dense/mul_2* 
_output_shapes
:
*
T0
Ù
Kbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
dtype0*
_output_shapes
: 
Î
Lbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/shape*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ý
Ibert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel
Ë
Ebert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel* 
_output_shapes
:

Ý
(bert/encoder/layer_4/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
	container *
shape:

»
/bert/encoder/layer_4/output/dense/kernel/AssignAssign(bert/encoder/layer_4/output/dense/kernelEbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
validate_shape(* 
_output_shapes
:

Ë
-bert/encoder/layer_4/output/dense/kernel/readIdentity(bert/encoder/layer_4/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel* 
_output_shapes
:

Â
8bert/encoder/layer_4/output/dense/bias/Initializer/zerosConst*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
dtype0*
_output_shapes	
:
Ï
&bert/encoder/layer_4/output/dense/bias
VariableV2*
shared_name *9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
£
-bert/encoder/layer_4/output/dense/bias/AssignAssign&bert/encoder/layer_4/output/dense/bias8bert/encoder/layer_4/output/dense/bias/Initializer/zeros*
T0*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
À
+bert/encoder/layer_4/output/dense/bias/readIdentity&bert/encoder/layer_4/output/dense/bias*
_output_shapes	
:*
T0*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias
á
(bert/encoder/layer_4/output/dense/MatMulMatMul-bert/encoder/layer_4/intermediate/dense/mul_3-bert/encoder/layer_4/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
Í
)bert/encoder/layer_4/output/dense/BiasAddBiasAdd(bert/encoder/layer_4/output/dense/MatMul+bert/encoder/layer_4/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

½
bert/encoder/layer_4/output/addAdd)bert/encoder/layer_4/output/dense/BiasAdd?bert/encoder/layer_4/attention/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
*
T0
Ê
<bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_4/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
	container *
shape:
³
1bert/encoder/layer_4/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_4/output/LayerNorm/beta<bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
Ì
/bert/encoder/layer_4/output/LayerNorm/beta/readIdentity*bert/encoder/layer_4/output/LayerNorm/beta*
_output_shapes	
:*
T0*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta
Ë
<bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Ù
+bert/encoder/layer_4/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
	container *
shape:
¶
2bert/encoder/layer_4/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_4/output/LayerNorm/gamma<bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_4/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_4/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
_output_shapes	
:

Dbert/encoder/layer_4/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
è
2bert/encoder/layer_4/output/LayerNorm/moments/meanMeanbert/encoder/layer_4/output/addDbert/encoder/layer_4/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
¨
:bert/encoder/layer_4/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_4/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_4/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_4/output/add:bert/encoder/layer_4/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
*
T0

Hbert/encoder/layer_4/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

6bert/encoder/layer_4/output/LayerNorm/moments/varianceMean?bert/encoder/layer_4/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_4/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
z
5bert/encoder/layer_4/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_4/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_4/output/LayerNorm/moments/variance5bert/encoder/layer_4/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0

5bert/encoder/layer_4/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_4/output/LayerNorm/batchnorm/add*
_output_shapes
:	*
T0
Î
3bert/encoder/layer_4/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_4/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

½
5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_4/output/add3bert/encoder/layer_4/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
Ð
5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_4/output/LayerNorm/moments/mean3bert/encoder/layer_4/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Í
3bert/encoder/layer_4/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_4/output/LayerNorm/beta/read5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ó
5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_4/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_4/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
é
Sbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
dtype0
Û
]bert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
seed2 
ý
Qbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel* 
_output_shapes
:

í
0bert/encoder/layer_5/attention/self/query/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Û
7bert/encoder/layer_5/attention/self/query/kernel/AssignAssign0bert/encoder/layer_5/attention/self/query/kernelMbert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel
ã
5bert/encoder/layer_5/attention/self/query/kernel/readIdentity0bert/encoder/layer_5/attention/self/query/kernel* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel
Ò
@bert/encoder/layer_5/attention/self/query/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_5/attention/self/query/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
	container *
shape:
Ã
5bert/encoder/layer_5/attention/self/query/bias/AssignAssign.bert/encoder/layer_5/attention/self/query/bias@bert/encoder/layer_5/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_5/attention/self/query/bias/readIdentity.bert/encoder/layer_5/attention/self/query/bias*
_output_shapes	
:*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias
ù
0bert/encoder/layer_5/attention/self/query/MatMulMatMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_15bert/encoder/layer_5/attention/self/query/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_5/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_5/attention/self/query/MatMul3bert/encoder/layer_5/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

å
Qbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
dtype0
Õ
[bert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
seed2 
õ
Obert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/stddev*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel* 
_output_shapes
:
*
T0
ã
Kbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel* 
_output_shapes
:

é
.bert/encoder/layer_5/attention/self/key/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ó
5bert/encoder/layer_5/attention/self/key/kernel/AssignAssign.bert/encoder/layer_5/attention/self/key/kernelKbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_5/attention/self/key/kernel/readIdentity.bert/encoder/layer_5/attention/self/key/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel
Î
>bert/encoder/layer_5/attention/self/key/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias
Û
,bert/encoder/layer_5/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
	container *
shape:
»
3bert/encoder/layer_5/attention/self/key/bias/AssignAssign,bert/encoder/layer_5/attention/self/key/bias>bert/encoder/layer_5/attention/self/key/bias/Initializer/zeros*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ò
1bert/encoder/layer_5/attention/self/key/bias/readIdentity,bert/encoder/layer_5/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
_output_shapes	
:
õ
.bert/encoder/layer_5/attention/self/key/MatMulMatMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_13bert/encoder/layer_5/attention/self/key/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ß
/bert/encoder/layer_5/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_5/attention/self/key/MatMul1bert/encoder/layer_5/attention/self/key/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC
é
Sbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
dtype0
Þ
Tbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
seed2 
ý
Qbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel
ë
Mbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel
í
0bert/encoder/layer_5/attention/self/value/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Û
7bert/encoder/layer_5/attention/self/value/kernel/AssignAssign0bert/encoder/layer_5/attention/self/value/kernelMbert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_5/attention/self/value/kernel/readIdentity0bert/encoder/layer_5/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_5/attention/self/value/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
dtype0
ß
.bert/encoder/layer_5/attention/self/value/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias
Ã
5bert/encoder/layer_5/attention/self/value/bias/AssignAssign.bert/encoder/layer_5/attention/self/value/bias@bert/encoder/layer_5/attention/self/value/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias
Ø
3bert/encoder/layer_5/attention/self/value/bias/readIdentity.bert/encoder/layer_5/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
_output_shapes	
:
ù
0bert/encoder/layer_5/attention/self/value/MatMulMatMul5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_15bert/encoder/layer_5/attention/self/value/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_5/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_5/attention/self/value/MatMul3bert/encoder/layer_5/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:


1bert/encoder/layer_5/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Ü
+bert/encoder/layer_5/attention/self/ReshapeReshape1bert/encoder/layer_5/attention/self/query/BiasAdd1bert/encoder/layer_5/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:@

2bert/encoder/layer_5/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_5/attention/self/transpose	Transpose+bert/encoder/layer_5/attention/self/Reshape2bert/encoder/layer_5/attention/self/transpose/perm*
Tperm0*
T0*'
_output_shapes
:@

3bert/encoder/layer_5/attention/self/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Þ
-bert/encoder/layer_5/attention/self/Reshape_1Reshape/bert/encoder/layer_5/attention/self/key/BiasAdd3bert/encoder/layer_5/attention/self/Reshape_1/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_5/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_5/attention/self/transpose_1	Transpose-bert/encoder/layer_5/attention/self/Reshape_14bert/encoder/layer_5/attention/self/transpose_1/perm*
T0*'
_output_shapes
:@*
Tperm0
è
*bert/encoder/layer_5/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_5/attention/self/transpose/bert/encoder/layer_5/attention/self/transpose_1*(
_output_shapes
:*
adj_x( *
adj_y(*
T0
n
)bert/encoder/layer_5/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
¸
'bert/encoder/layer_5/attention/self/MulMul*bert/encoder/layer_5/attention/self/MatMul)bert/encoder/layer_5/attention/self/Mul/y*
T0*(
_output_shapes
:
|
2bert/encoder/layer_5/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Á
.bert/encoder/layer_5/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_5/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:
n
)bert/encoder/layer_5/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¼
'bert/encoder/layer_5/attention/self/subSub)bert/encoder/layer_5/attention/self/sub/x.bert/encoder/layer_5/attention/self/ExpandDims*
T0*(
_output_shapes
:
p
+bert/encoder/layer_5/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¹
)bert/encoder/layer_5/attention/self/mul_1Mul'bert/encoder/layer_5/attention/self/sub+bert/encoder/layer_5/attention/self/mul_1/y*
T0*(
_output_shapes
:
µ
'bert/encoder/layer_5/attention/self/addAdd'bert/encoder/layer_5/attention/self/Mul)bert/encoder/layer_5/attention/self/mul_1*(
_output_shapes
:*
T0

+bert/encoder/layer_5/attention/self/SoftmaxSoftmax'bert/encoder/layer_5/attention/self/add*
T0*(
_output_shapes
:

3bert/encoder/layer_5/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
à
-bert/encoder/layer_5/attention/self/Reshape_2Reshape1bert/encoder/layer_5/attention/self/value/BiasAdd3bert/encoder/layer_5/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_5/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_5/attention/self/transpose_2	Transpose-bert/encoder/layer_5/attention/self/Reshape_24bert/encoder/layer_5/attention/self/transpose_2/perm*
T0*'
_output_shapes
:@*
Tperm0
ç
,bert/encoder/layer_5/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_5/attention/self/Softmax/bert/encoder/layer_5/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:@

4bert/encoder/layer_5/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ß
/bert/encoder/layer_5/attention/self/transpose_3	Transpose,bert/encoder/layer_5/attention/self/MatMul_14bert/encoder/layer_5/attention/self/transpose_3/perm*
T0*'
_output_shapes
:@*
Tperm0

3bert/encoder/layer_5/attention/self/Reshape_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
×
-bert/encoder/layer_5/attention/self/Reshape_3Reshape/bert/encoder/layer_5/attention/self/transpose_33bert/encoder/layer_5/attention/self/Reshape_3/shape* 
_output_shapes
:
*
T0*
Tshape0
í
Ubert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
dtype0*
_output_shapes
:
à
Tbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
â
Vbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel
á
_bert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
seed2 

Sbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel* 
_output_shapes
:

ñ
2bert/encoder/layer_5/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
	container *
shape:

ã
9bert/encoder/layer_5/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_5/attention/output/dense/kernelObert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
é
7bert/encoder/layer_5/attention/output/dense/kernel/readIdentity2bert/encoder/layer_5/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel* 
_output_shapes
:

Ö
Bbert/encoder/layer_5/attention/output/dense/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
dtype0
ã
0bert/encoder/layer_5/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
	container *
shape:
Ë
7bert/encoder/layer_5/attention/output/dense/bias/AssignAssign0bert/encoder/layer_5/attention/output/dense/biasBbert/encoder/layer_5/attention/output/dense/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
validate_shape(
Þ
5bert/encoder/layer_5/attention/output/dense/bias/readIdentity0bert/encoder/layer_5/attention/output/dense/bias*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
_output_shapes	
:*
T0
õ
2bert/encoder/layer_5/attention/output/dense/MatMulMatMul-bert/encoder/layer_5/attention/self/Reshape_37bert/encoder/layer_5/attention/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ë
3bert/encoder/layer_5/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_5/attention/output/dense/MatMul5bert/encoder/layer_5/attention/output/dense/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC
Ç
)bert/encoder/layer_5/attention/output/addAdd3bert/encoder/layer_5/attention/output/dense/BiasAdd5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Þ
Fbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta
ë
4bert/encoder/layer_5/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
	container *
shape:
Û
;bert/encoder/layer_5/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_5/attention/output/LayerNorm/betaFbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ê
9bert/encoder/layer_5/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_5/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
_output_shapes	
:
ß
Fbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_5/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
	container *
shape:
Þ
<bert/encoder/layer_5/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_5/attention/output/LayerNorm/gammaFbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
validate_shape(
í
:bert/encoder/layer_5/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_5/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
_output_shapes	
:

Nbert/encoder/layer_5/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

<bert/encoder/layer_5/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_5/attention/output/addNbert/encoder/layer_5/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¼
Dbert/encoder/layer_5/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_5/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ú
Ibert/encoder/layer_5/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_5/attention/output/addDbert/encoder/layer_5/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Rbert/encoder/layer_5/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_5/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_5/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_5/attention/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	

?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_5/attention/output/LayerNorm/moments/variance?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0
±
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ì
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_5/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Û
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_5/attention/output/add=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_5/attention/output/LayerNorm/moments/mean=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
ë
=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_5/attention/output/LayerNorm/beta/read?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ñ
?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
å
Qbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/shape*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0
õ
Obert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel* 
_output_shapes
:

é
.bert/encoder/layer_5/intermediate/dense/kernel
VariableV2*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ó
5bert/encoder/layer_5/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_5/intermediate/dense/kernelKbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ý
3bert/encoder/layer_5/intermediate/dense/kernel/readIdentity.bert/encoder/layer_5/intermediate/dense/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel
Ú
Nbert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ê
Dbert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
dtype0
Õ
>bert/encoder/layer_5/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros/Const*
T0*

index_type0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
_output_shapes	
:
Û
,bert/encoder/layer_5/intermediate/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
	container 
»
3bert/encoder/layer_5/intermediate/dense/bias/AssignAssign,bert/encoder/layer_5/intermediate/dense/bias>bert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias
Ò
1bert/encoder/layer_5/intermediate/dense/bias/readIdentity,bert/encoder/layer_5/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
_output_shapes	
:
ÿ
.bert/encoder/layer_5/intermediate/dense/MatMulMatMul?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_5/intermediate/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
ß
/bert/encoder/layer_5/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_5/intermediate/dense/MatMul1bert/encoder/layer_5/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_5/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
½
+bert/encoder/layer_5/intermediate/dense/PowPow/bert/encoder/layer_5/intermediate/dense/BiasAdd-bert/encoder/layer_5/intermediate/dense/Pow/y* 
_output_shapes
:
*
T0
r
-bert/encoder/layer_5/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
¹
+bert/encoder/layer_5/intermediate/dense/mulMul-bert/encoder/layer_5/intermediate/dense/mul/x+bert/encoder/layer_5/intermediate/dense/Pow*
T0* 
_output_shapes
:

»
+bert/encoder/layer_5/intermediate/dense/addAdd/bert/encoder/layer_5/intermediate/dense/BiasAdd+bert/encoder/layer_5/intermediate/dense/mul*
T0* 
_output_shapes
:

t
/bert/encoder/layer_5/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
½
-bert/encoder/layer_5/intermediate/dense/mul_1Mul/bert/encoder/layer_5/intermediate/dense/mul_1/x+bert/encoder/layer_5/intermediate/dense/add* 
_output_shapes
:
*
T0

,bert/encoder/layer_5/intermediate/dense/TanhTanh-bert/encoder/layer_5/intermediate/dense/mul_1*
T0* 
_output_shapes
:

t
/bert/encoder/layer_5/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¾
-bert/encoder/layer_5/intermediate/dense/add_1Add/bert/encoder/layer_5/intermediate/dense/add_1/x,bert/encoder/layer_5/intermediate/dense/Tanh*
T0* 
_output_shapes
:

t
/bert/encoder/layer_5/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
¿
-bert/encoder/layer_5/intermediate/dense/mul_2Mul/bert/encoder/layer_5/intermediate/dense/mul_2/x-bert/encoder/layer_5/intermediate/dense/add_1*
T0* 
_output_shapes
:

¿
-bert/encoder/layer_5/intermediate/dense/mul_3Mul/bert/encoder/layer_5/intermediate/dense/BiasAdd-bert/encoder/layer_5/intermediate/dense/mul_2*
T0* 
_output_shapes
:

Ù
Kbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
dtype0*
_output_shapes
: 
Î
Lbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel
Ý
Ibert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel* 
_output_shapes
:

Ý
(bert/encoder/layer_5/output/dense/kernel
VariableV2*
shared_name *;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

»
/bert/encoder/layer_5/output/dense/kernel/AssignAssign(bert/encoder/layer_5/output/dense/kernelEbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
validate_shape(
Ë
-bert/encoder/layer_5/output/dense/kernel/readIdentity(bert/encoder/layer_5/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel* 
_output_shapes
:

Â
8bert/encoder/layer_5/output/dense/bias/Initializer/zerosConst*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
dtype0*
_output_shapes	
:
Ï
&bert/encoder/layer_5/output/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias
£
-bert/encoder/layer_5/output/dense/bias/AssignAssign&bert/encoder/layer_5/output/dense/bias8bert/encoder/layer_5/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
validate_shape(*
_output_shapes	
:
À
+bert/encoder/layer_5/output/dense/bias/readIdentity&bert/encoder/layer_5/output/dense/bias*
_output_shapes	
:*
T0*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias
á
(bert/encoder/layer_5/output/dense/MatMulMatMul-bert/encoder/layer_5/intermediate/dense/mul_3-bert/encoder/layer_5/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
Í
)bert/encoder/layer_5/output/dense/BiasAddBiasAdd(bert/encoder/layer_5/output/dense/MatMul+bert/encoder/layer_5/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

½
bert/encoder/layer_5/output/addAdd)bert/encoder/layer_5/output/dense/BiasAdd?bert/encoder/layer_5/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ê
<bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_5/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
	container *
shape:
³
1bert/encoder/layer_5/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_5/output/LayerNorm/beta<bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
Ì
/bert/encoder/layer_5/output/LayerNorm/beta/readIdentity*bert/encoder/layer_5/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
_output_shapes	
:
Ë
<bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Ù
+bert/encoder/layer_5/output/LayerNorm/gamma
VariableV2*
shared_name *>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
¶
2bert/encoder/layer_5/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_5/output/LayerNorm/gamma<bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
validate_shape(
Ï
0bert/encoder/layer_5/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_5/output/LayerNorm/gamma*
_output_shapes	
:*
T0*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma

Dbert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
è
2bert/encoder/layer_5/output/LayerNorm/moments/meanMeanbert/encoder/layer_5/output/addDbert/encoder/layer_5/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¨
:bert/encoder/layer_5/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_5/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_5/output/add:bert/encoder/layer_5/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Hbert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

6bert/encoder/layer_5/output/LayerNorm/moments/varianceMean?bert/encoder/layer_5/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_5/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
z
5bert/encoder/layer_5/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_5/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_5/output/LayerNorm/moments/variance5bert/encoder/layer_5/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0

5bert/encoder/layer_5/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_5/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
Î
3bert/encoder/layer_5/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_5/output/LayerNorm/gamma/read* 
_output_shapes
:
*
T0
½
5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_5/output/add3bert/encoder/layer_5/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_5/output/LayerNorm/moments/mean3bert/encoder/layer_5/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
Í
3bert/encoder/layer_5/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_5/output/LayerNorm/beta/read5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ó
5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_5/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_5/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

é
Sbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/shape*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:

ý
Qbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/stddev*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel* 
_output_shapes
:
*
T0
ë
Mbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel
í
0bert/encoder/layer_6/attention/self/query/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Û
7bert/encoder/layer_6/attention/self/query/kernel/AssignAssign0bert/encoder/layer_6/attention/self/query/kernelMbert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_6/attention/self/query/kernel/readIdentity0bert/encoder/layer_6/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_6/attention/self/query/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_6/attention/self/query/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
	container *
shape:
Ã
5bert/encoder/layer_6/attention/self/query/bias/AssignAssign.bert/encoder/layer_6/attention/self/query/bias@bert/encoder/layer_6/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_6/attention/self/query/bias/readIdentity.bert/encoder/layer_6/attention/self/query/bias*
_output_shapes	
:*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias
ù
0bert/encoder/layer_6/attention/self/query/MatMulMatMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_15bert/encoder/layer_6/attention/self/query/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_6/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_6/attention/self/query/MatMul3bert/encoder/layer_6/attention/self/query/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
å
Qbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
õ
Obert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel
ã
Kbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel* 
_output_shapes
:

é
.bert/encoder/layer_6/attention/self/key/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
	container 
Ó
5bert/encoder/layer_6/attention/self/key/kernel/AssignAssign.bert/encoder/layer_6/attention/self/key/kernelKbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_6/attention/self/key/kernel/readIdentity.bert/encoder/layer_6/attention/self/key/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel* 
_output_shapes
:

Î
>bert/encoder/layer_6/attention/self/key/bias/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
dtype0*
_output_shapes	
:
Û
,bert/encoder/layer_6/attention/self/key/bias
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
»
3bert/encoder/layer_6/attention/self/key/bias/AssignAssign,bert/encoder/layer_6/attention/self/key/bias>bert/encoder/layer_6/attention/self/key/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias
Ò
1bert/encoder/layer_6/attention/self/key/bias/readIdentity,bert/encoder/layer_6/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
_output_shapes	
:
õ
.bert/encoder/layer_6/attention/self/key/MatMulMatMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_13bert/encoder/layer_6/attention/self/key/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ß
/bert/encoder/layer_6/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_6/attention/self/key/MatMul1bert/encoder/layer_6/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

é
Sbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel
Þ
Tbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel
Û
]bert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
seed2 
ý
Qbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel
í
0bert/encoder/layer_6/attention/self/value/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Û
7bert/encoder/layer_6/attention/self/value/kernel/AssignAssign0bert/encoder/layer_6/attention/self/value/kernelMbert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
validate_shape(
ã
5bert/encoder/layer_6/attention/self/value/kernel/readIdentity0bert/encoder/layer_6/attention/self/value/kernel* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel
Ò
@bert/encoder/layer_6/attention/self/value/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_6/attention/self/value/bias
VariableV2*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ã
5bert/encoder/layer_6/attention/self/value/bias/AssignAssign.bert/encoder/layer_6/attention/self/value/bias@bert/encoder/layer_6/attention/self/value/bias/Initializer/zeros*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ø
3bert/encoder/layer_6/attention/self/value/bias/readIdentity.bert/encoder/layer_6/attention/self/value/bias*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
_output_shapes	
:
ù
0bert/encoder/layer_6/attention/self/value/MatMulMatMul5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_15bert/encoder/layer_6/attention/self/value/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
å
1bert/encoder/layer_6/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_6/attention/self/value/MatMul3bert/encoder/layer_6/attention/self/value/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC

1bert/encoder/layer_6/attention/self/Reshape/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
Ü
+bert/encoder/layer_6/attention/self/ReshapeReshape1bert/encoder/layer_6/attention/self/query/BiasAdd1bert/encoder/layer_6/attention/self/Reshape/shape*
Tshape0*'
_output_shapes
:@*
T0

2bert/encoder/layer_6/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_6/attention/self/transpose	Transpose+bert/encoder/layer_6/attention/self/Reshape2bert/encoder/layer_6/attention/self/transpose/perm*
T0*'
_output_shapes
:@*
Tperm0

3bert/encoder/layer_6/attention/self/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Þ
-bert/encoder/layer_6/attention/self/Reshape_1Reshape/bert/encoder/layer_6/attention/self/key/BiasAdd3bert/encoder/layer_6/attention/self/Reshape_1/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_6/attention/self/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
à
/bert/encoder/layer_6/attention/self/transpose_1	Transpose-bert/encoder/layer_6/attention/self/Reshape_14bert/encoder/layer_6/attention/self/transpose_1/perm*'
_output_shapes
:@*
Tperm0*
T0
è
*bert/encoder/layer_6/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_6/attention/self/transpose/bert/encoder/layer_6/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:
n
)bert/encoder/layer_6/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
¸
'bert/encoder/layer_6/attention/self/MulMul*bert/encoder/layer_6/attention/self/MatMul)bert/encoder/layer_6/attention/self/Mul/y*
T0*(
_output_shapes
:
|
2bert/encoder/layer_6/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Á
.bert/encoder/layer_6/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_6/attention/self/ExpandDims/dim*
T0*(
_output_shapes
:*

Tdim0
n
)bert/encoder/layer_6/attention/self/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
¼
'bert/encoder/layer_6/attention/self/subSub)bert/encoder/layer_6/attention/self/sub/x.bert/encoder/layer_6/attention/self/ExpandDims*(
_output_shapes
:*
T0
p
+bert/encoder/layer_6/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¹
)bert/encoder/layer_6/attention/self/mul_1Mul'bert/encoder/layer_6/attention/self/sub+bert/encoder/layer_6/attention/self/mul_1/y*
T0*(
_output_shapes
:
µ
'bert/encoder/layer_6/attention/self/addAdd'bert/encoder/layer_6/attention/self/Mul)bert/encoder/layer_6/attention/self/mul_1*(
_output_shapes
:*
T0

+bert/encoder/layer_6/attention/self/SoftmaxSoftmax'bert/encoder/layer_6/attention/self/add*
T0*(
_output_shapes
:

3bert/encoder/layer_6/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
à
-bert/encoder/layer_6/attention/self/Reshape_2Reshape1bert/encoder/layer_6/attention/self/value/BiasAdd3bert/encoder/layer_6/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_6/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_6/attention/self/transpose_2	Transpose-bert/encoder/layer_6/attention/self/Reshape_24bert/encoder/layer_6/attention/self/transpose_2/perm*
T0*'
_output_shapes
:@*
Tperm0
ç
,bert/encoder/layer_6/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_6/attention/self/Softmax/bert/encoder/layer_6/attention/self/transpose_2*'
_output_shapes
:@*
adj_x( *
adj_y( *
T0

4bert/encoder/layer_6/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ß
/bert/encoder/layer_6/attention/self/transpose_3	Transpose,bert/encoder/layer_6/attention/self/MatMul_14bert/encoder/layer_6/attention/self/transpose_3/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_6/attention/self/Reshape_3/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
×
-bert/encoder/layer_6/attention/self/Reshape_3Reshape/bert/encoder/layer_6/attention/self/transpose_33bert/encoder/layer_6/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:

í
Ubert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
dtype0*
_output_shapes
:
à
Tbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
dtype0
â
Vbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
á
_bert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
seed2 

Sbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel
ñ
2bert/encoder/layer_6/attention/output/dense/kernel
VariableV2*
shared_name *E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ã
9bert/encoder/layer_6/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_6/attention/output/dense/kernelObert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
é
7bert/encoder/layer_6/attention/output/dense/kernel/readIdentity2bert/encoder/layer_6/attention/output/dense/kernel* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel
Ö
Bbert/encoder/layer_6/attention/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias
ã
0bert/encoder/layer_6/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
	container *
shape:
Ë
7bert/encoder/layer_6/attention/output/dense/bias/AssignAssign0bert/encoder/layer_6/attention/output/dense/biasBbert/encoder/layer_6/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
Þ
5bert/encoder/layer_6/attention/output/dense/bias/readIdentity0bert/encoder/layer_6/attention/output/dense/bias*
_output_shapes	
:*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias
õ
2bert/encoder/layer_6/attention/output/dense/MatMulMatMul-bert/encoder/layer_6/attention/self/Reshape_37bert/encoder/layer_6/attention/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ë
3bert/encoder/layer_6/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_6/attention/output/dense/MatMul5bert/encoder/layer_6/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

Ç
)bert/encoder/layer_6/attention/output/addAdd3bert/encoder/layer_6/attention/output/dense/BiasAdd5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Þ
Fbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
ë
4bert/encoder/layer_6/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
	container *
shape:
Û
;bert/encoder/layer_6/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_6/attention/output/LayerNorm/betaFbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ê
9bert/encoder/layer_6/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_6/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
_output_shapes	
:
ß
Fbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma
í
5bert/encoder/layer_6/attention/output/LayerNorm/gamma
VariableV2*
shared_name *H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
Þ
<bert/encoder/layer_6/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_6/attention/output/LayerNorm/gammaFbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/ones*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma
í
:bert/encoder/layer_6/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_6/attention/output/LayerNorm/gamma*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma*
_output_shapes	
:*
T0

Nbert/encoder/layer_6/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

<bert/encoder/layer_6/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_6/attention/output/addNbert/encoder/layer_6/attention/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
¼
Dbert/encoder/layer_6/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_6/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ú
Ibert/encoder/layer_6/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_6/attention/output/addDbert/encoder/layer_6/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Rbert/encoder/layer_6/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
®
@bert/encoder/layer_6/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_6/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_6/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0

?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_6/attention/output/LayerNorm/moments/variance?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	
±
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ì
=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_6/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Û
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_6/attention/output/add=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
î
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_6/attention/output/LayerNorm/moments/mean=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
ë
=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_6/attention/output/LayerNorm/beta/read?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ñ
?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

å
Qbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
õ
Obert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel* 
_output_shapes
:

é
.bert/encoder/layer_6/intermediate/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
	container *
shape:

Ó
5bert/encoder/layer_6/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_6/intermediate/dense/kernelKbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ý
3bert/encoder/layer_6/intermediate/dense/kernel/readIdentity.bert/encoder/layer_6/intermediate/dense/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel
Ú
Nbert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB:*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias
Ê
Dbert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
dtype0*
_output_shapes
: 
Õ
>bert/encoder/layer_6/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros/Const*

index_type0*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
_output_shapes	
:*
T0
Û
,bert/encoder/layer_6/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
	container *
shape:
»
3bert/encoder/layer_6/intermediate/dense/bias/AssignAssign,bert/encoder/layer_6/intermediate/dense/bias>bert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ò
1bert/encoder/layer_6/intermediate/dense/bias/readIdentity,bert/encoder/layer_6/intermediate/dense/bias*
_output_shapes	
:*
T0*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias
ÿ
.bert/encoder/layer_6/intermediate/dense/MatMulMatMul?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_6/intermediate/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ß
/bert/encoder/layer_6/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_6/intermediate/dense/MatMul1bert/encoder/layer_6/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_6/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
½
+bert/encoder/layer_6/intermediate/dense/PowPow/bert/encoder/layer_6/intermediate/dense/BiasAdd-bert/encoder/layer_6/intermediate/dense/Pow/y*
T0* 
_output_shapes
:

r
-bert/encoder/layer_6/intermediate/dense/mul/xConst*
dtype0*
_output_shapes
: *
valueB
 *'7=
¹
+bert/encoder/layer_6/intermediate/dense/mulMul-bert/encoder/layer_6/intermediate/dense/mul/x+bert/encoder/layer_6/intermediate/dense/Pow* 
_output_shapes
:
*
T0
»
+bert/encoder/layer_6/intermediate/dense/addAdd/bert/encoder/layer_6/intermediate/dense/BiasAdd+bert/encoder/layer_6/intermediate/dense/mul* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_6/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
½
-bert/encoder/layer_6/intermediate/dense/mul_1Mul/bert/encoder/layer_6/intermediate/dense/mul_1/x+bert/encoder/layer_6/intermediate/dense/add*
T0* 
_output_shapes
:


,bert/encoder/layer_6/intermediate/dense/TanhTanh-bert/encoder/layer_6/intermediate/dense/mul_1*
T0* 
_output_shapes
:

t
/bert/encoder/layer_6/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¾
-bert/encoder/layer_6/intermediate/dense/add_1Add/bert/encoder/layer_6/intermediate/dense/add_1/x,bert/encoder/layer_6/intermediate/dense/Tanh* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_6/intermediate/dense/mul_2/xConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
¿
-bert/encoder/layer_6/intermediate/dense/mul_2Mul/bert/encoder/layer_6/intermediate/dense/mul_2/x-bert/encoder/layer_6/intermediate/dense/add_1*
T0* 
_output_shapes
:

¿
-bert/encoder/layer_6/intermediate/dense/mul_3Mul/bert/encoder/layer_6/intermediate/dense/BiasAdd-bert/encoder/layer_6/intermediate/dense/mul_2* 
_output_shapes
:
*
T0
Ù
Kbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
dtype0*
_output_shapes
: 
Î
Lbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel
Ý
Ibert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal/mean*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel* 
_output_shapes
:
*
T0
Ý
(bert/encoder/layer_6/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
	container *
shape:

»
/bert/encoder/layer_6/output/dense/kernel/AssignAssign(bert/encoder/layer_6/output/dense/kernelEbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ë
-bert/encoder/layer_6/output/dense/kernel/readIdentity(bert/encoder/layer_6/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel* 
_output_shapes
:

Â
8bert/encoder/layer_6/output/dense/bias/Initializer/zerosConst*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
dtype0*
_output_shapes	
:
Ï
&bert/encoder/layer_6/output/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias
£
-bert/encoder/layer_6/output/dense/bias/AssignAssign&bert/encoder/layer_6/output/dense/bias8bert/encoder/layer_6/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
validate_shape(*
_output_shapes	
:
À
+bert/encoder/layer_6/output/dense/bias/readIdentity&bert/encoder/layer_6/output/dense/bias*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
_output_shapes	
:*
T0
á
(bert/encoder/layer_6/output/dense/MatMulMatMul-bert/encoder/layer_6/intermediate/dense/mul_3-bert/encoder/layer_6/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
Í
)bert/encoder/layer_6/output/dense/BiasAddBiasAdd(bert/encoder/layer_6/output/dense/MatMul+bert/encoder/layer_6/output/dense/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
½
bert/encoder/layer_6/output/addAdd)bert/encoder/layer_6/output/dense/BiasAdd?bert/encoder/layer_6/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ê
<bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
dtype0
×
*bert/encoder/layer_6/output/LayerNorm/beta
VariableV2*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
³
1bert/encoder/layer_6/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_6/output/LayerNorm/beta<bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
Ì
/bert/encoder/layer_6/output/LayerNorm/beta/readIdentity*bert/encoder/layer_6/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
_output_shapes	
:
Ë
<bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/onesConst*
_output_shapes	
:*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
dtype0
Ù
+bert/encoder/layer_6/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
	container *
shape:
¶
2bert/encoder/layer_6/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_6/output/LayerNorm/gamma<bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_6/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_6/output/LayerNorm/gamma*
_output_shapes	
:*
T0*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma

Dbert/encoder/layer_6/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
è
2bert/encoder/layer_6/output/LayerNorm/moments/meanMeanbert/encoder/layer_6/output/addDbert/encoder/layer_6/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¨
:bert/encoder/layer_6/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_6/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_6/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_6/output/add:bert/encoder/layer_6/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Hbert/encoder/layer_6/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

6bert/encoder/layer_6/output/LayerNorm/moments/varianceMean?bert/encoder/layer_6/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_6/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
z
5bert/encoder/layer_6/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_6/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_6/output/LayerNorm/moments/variance5bert/encoder/layer_6/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0

5bert/encoder/layer_6/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_6/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
Î
3bert/encoder/layer_6/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_6/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_6/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

½
5bert/encoder/layer_6/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_6/output/add3bert/encoder/layer_6/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
5bert/encoder/layer_6/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_6/output/LayerNorm/moments/mean3bert/encoder/layer_6/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
Í
3bert/encoder/layer_6/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_6/output/LayerNorm/beta/read5bert/encoder/layer_6/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ó
5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_6/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_6/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
é
Sbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel
Û
]bert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel
ý
Qbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel
ë
Mbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel* 
_output_shapes
:

í
0bert/encoder/layer_7/attention/self/query/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Û
7bert/encoder/layer_7/attention/self/query/kernel/AssignAssign0bert/encoder/layer_7/attention/self/query/kernelMbert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ã
5bert/encoder/layer_7/attention/self/query/kernel/readIdentity0bert/encoder/layer_7/attention/self/query/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_7/attention/self/query/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_7/attention/self/query/bias
VariableV2*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ã
5bert/encoder/layer_7/attention/self/query/bias/AssignAssign.bert/encoder/layer_7/attention/self/query/bias@bert/encoder/layer_7/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_7/attention/self/query/bias/readIdentity.bert/encoder/layer_7/attention/self/query/bias*
_output_shapes	
:*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias
ù
0bert/encoder/layer_7/attention/self/query/MatMulMatMul5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_15bert/encoder/layer_7/attention/self/query/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_7/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_7/attention/self/query/MatMul3bert/encoder/layer_7/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

å
Qbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel
õ
Obert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel
é
.bert/encoder/layer_7/attention/self/key/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
	container *
shape:

Ó
5bert/encoder/layer_7/attention/self/key/kernel/AssignAssign.bert/encoder/layer_7/attention/self/key/kernelKbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ý
3bert/encoder/layer_7/attention/self/key/kernel/readIdentity.bert/encoder/layer_7/attention/self/key/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel
Î
>bert/encoder/layer_7/attention/self/key/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
dtype0
Û
,bert/encoder/layer_7/attention/self/key/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias
»
3bert/encoder/layer_7/attention/self/key/bias/AssignAssign,bert/encoder/layer_7/attention/self/key/bias>bert/encoder/layer_7/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_7/attention/self/key/bias/readIdentity,bert/encoder/layer_7/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
_output_shapes	
:
õ
.bert/encoder/layer_7/attention/self/key/MatMulMatMul5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_13bert/encoder/layer_7/attention/self/key/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ß
/bert/encoder/layer_7/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_7/attention/self/key/MatMul1bert/encoder/layer_7/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

é
Sbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
dtype0
Ü
Rbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
dtype0
Þ
Tbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
seed2 
ý
Qbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel
ë
Mbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel* 
_output_shapes
:

í
0bert/encoder/layer_7/attention/self/value/kernel
VariableV2*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Û
7bert/encoder/layer_7/attention/self/value/kernel/AssignAssign0bert/encoder/layer_7/attention/self/value/kernelMbert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ã
5bert/encoder/layer_7/attention/self/value/kernel/readIdentity0bert/encoder/layer_7/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_7/attention/self/value/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_7/attention/self/value/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ã
5bert/encoder/layer_7/attention/self/value/bias/AssignAssign.bert/encoder/layer_7/attention/self/value/bias@bert/encoder/layer_7/attention/self/value/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
validate_shape(
Ø
3bert/encoder/layer_7/attention/self/value/bias/readIdentity.bert/encoder/layer_7/attention/self/value/bias*
_output_shapes	
:*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias
ù
0bert/encoder/layer_7/attention/self/value/MatMulMatMul5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_15bert/encoder/layer_7/attention/self/value/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_7/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_7/attention/self/value/MatMul3bert/encoder/layer_7/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:


1bert/encoder/layer_7/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Ü
+bert/encoder/layer_7/attention/self/ReshapeReshape1bert/encoder/layer_7/attention/self/query/BiasAdd1bert/encoder/layer_7/attention/self/Reshape/shape*'
_output_shapes
:@*
T0*
Tshape0

2bert/encoder/layer_7/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_7/attention/self/transpose	Transpose+bert/encoder/layer_7/attention/self/Reshape2bert/encoder/layer_7/attention/self/transpose/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_7/attention/self/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
Þ
-bert/encoder/layer_7/attention/self/Reshape_1Reshape/bert/encoder/layer_7/attention/self/key/BiasAdd3bert/encoder/layer_7/attention/self/Reshape_1/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_7/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_7/attention/self/transpose_1	Transpose-bert/encoder/layer_7/attention/self/Reshape_14bert/encoder/layer_7/attention/self/transpose_1/perm*'
_output_shapes
:@*
Tperm0*
T0
è
*bert/encoder/layer_7/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_7/attention/self/transpose/bert/encoder/layer_7/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:
n
)bert/encoder/layer_7/attention/self/Mul/yConst*
_output_shapes
: *
valueB
 *   >*
dtype0
¸
'bert/encoder/layer_7/attention/self/MulMul*bert/encoder/layer_7/attention/self/MatMul)bert/encoder/layer_7/attention/self/Mul/y*(
_output_shapes
:*
T0
|
2bert/encoder/layer_7/attention/self/ExpandDims/dimConst*
_output_shapes
:*
valueB:*
dtype0
Á
.bert/encoder/layer_7/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_7/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:
n
)bert/encoder/layer_7/attention/self/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ?
¼
'bert/encoder/layer_7/attention/self/subSub)bert/encoder/layer_7/attention/self/sub/x.bert/encoder/layer_7/attention/self/ExpandDims*
T0*(
_output_shapes
:
p
+bert/encoder/layer_7/attention/self/mul_1/yConst*
_output_shapes
: *
valueB
 * @Æ*
dtype0
¹
)bert/encoder/layer_7/attention/self/mul_1Mul'bert/encoder/layer_7/attention/self/sub+bert/encoder/layer_7/attention/self/mul_1/y*
T0*(
_output_shapes
:
µ
'bert/encoder/layer_7/attention/self/addAdd'bert/encoder/layer_7/attention/self/Mul)bert/encoder/layer_7/attention/self/mul_1*(
_output_shapes
:*
T0

+bert/encoder/layer_7/attention/self/SoftmaxSoftmax'bert/encoder/layer_7/attention/self/add*
T0*(
_output_shapes
:

3bert/encoder/layer_7/attention/self/Reshape_2/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
à
-bert/encoder/layer_7/attention/self/Reshape_2Reshape1bert/encoder/layer_7/attention/self/value/BiasAdd3bert/encoder/layer_7/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_7/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_7/attention/self/transpose_2	Transpose-bert/encoder/layer_7/attention/self/Reshape_24bert/encoder/layer_7/attention/self/transpose_2/perm*'
_output_shapes
:@*
Tperm0*
T0
ç
,bert/encoder/layer_7/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_7/attention/self/Softmax/bert/encoder/layer_7/attention/self/transpose_2*'
_output_shapes
:@*
adj_x( *
adj_y( *
T0

4bert/encoder/layer_7/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ß
/bert/encoder/layer_7/attention/self/transpose_3	Transpose,bert/encoder/layer_7/attention/self/MatMul_14bert/encoder/layer_7/attention/self/transpose_3/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_7/attention/self/Reshape_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
×
-bert/encoder/layer_7/attention/self/Reshape_3Reshape/bert/encoder/layer_7/attention/self/transpose_33bert/encoder/layer_7/attention/self/Reshape_3/shape* 
_output_shapes
:
*
T0*
Tshape0
í
Ubert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
dtype0*
_output_shapes
:
à
Tbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
â
Vbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
á
_bert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 

Sbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel
ñ
2bert/encoder/layer_7/attention/output/dense/kernel
VariableV2*
shared_name *E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ã
9bert/encoder/layer_7/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_7/attention/output/dense/kernelObert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

é
7bert/encoder/layer_7/attention/output/dense/kernel/readIdentity2bert/encoder/layer_7/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel* 
_output_shapes
:

Ö
Bbert/encoder/layer_7/attention/output/dense/bias/Initializer/zerosConst*
valueB*    *C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
dtype0*
_output_shapes	
:
ã
0bert/encoder/layer_7/attention/output/dense/bias
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ë
7bert/encoder/layer_7/attention/output/dense/bias/AssignAssign0bert/encoder/layer_7/attention/output/dense/biasBbert/encoder/layer_7/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
Þ
5bert/encoder/layer_7/attention/output/dense/bias/readIdentity0bert/encoder/layer_7/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
_output_shapes	
:
õ
2bert/encoder/layer_7/attention/output/dense/MatMulMatMul-bert/encoder/layer_7/attention/self/Reshape_37bert/encoder/layer_7/attention/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ë
3bert/encoder/layer_7/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_7/attention/output/dense/MatMul5bert/encoder/layer_7/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

Ç
)bert/encoder/layer_7/attention/output/addAdd3bert/encoder/layer_7/attention/output/dense/BiasAdd5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Þ
Fbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta
ë
4bert/encoder/layer_7/attention/output/LayerNorm/beta
VariableV2*
shared_name *G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
Û
;bert/encoder/layer_7/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_7/attention/output/LayerNorm/betaFbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta
ê
9bert/encoder/layer_7/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_7/attention/output/LayerNorm/beta*
T0*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
_output_shapes	
:
ß
Fbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_7/attention/output/LayerNorm/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma
Þ
<bert/encoder/layer_7/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_7/attention/output/LayerNorm/gammaFbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/ones*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
validate_shape(
í
:bert/encoder/layer_7/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_7/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
_output_shapes	
:

Nbert/encoder/layer_7/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

<bert/encoder/layer_7/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_7/attention/output/addNbert/encoder/layer_7/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
¼
Dbert/encoder/layer_7/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_7/attention/output/LayerNorm/moments/mean*
_output_shapes
:	*
T0
ú
Ibert/encoder/layer_7/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_7/attention/output/addDbert/encoder/layer_7/attention/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
*
T0

Rbert/encoder/layer_7/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_7/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_7/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_7/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0

?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_7/attention/output/LayerNorm/moments/variance?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	
±
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add*
_output_shapes
:	*
T0
ì
=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_7/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Û
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_7/attention/output/add=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_7/attention/output/LayerNorm/moments/mean=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
ë
=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_7/attention/output/LayerNorm/beta/read?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ñ
?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
å
Qbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
seed2 
õ
Obert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel
ã
Kbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel* 
_output_shapes
:

é
.bert/encoder/layer_7/intermediate/dense/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ó
5bert/encoder/layer_7/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_7/intermediate/dense/kernelKbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_7/intermediate/dense/kernel/readIdentity.bert/encoder/layer_7/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel* 
_output_shapes
:

Ú
Nbert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ê
Dbert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias
Õ
>bert/encoder/layer_7/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros/Const*
T0*

index_type0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
_output_shapes	
:
Û
,bert/encoder/layer_7/intermediate/dense/bias
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
»
3bert/encoder/layer_7/intermediate/dense/bias/AssignAssign,bert/encoder/layer_7/intermediate/dense/bias>bert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_7/intermediate/dense/bias/readIdentity,bert/encoder/layer_7/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
_output_shapes	
:
ÿ
.bert/encoder/layer_7/intermediate/dense/MatMulMatMul?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_7/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ß
/bert/encoder/layer_7/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_7/intermediate/dense/MatMul1bert/encoder/layer_7/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_7/intermediate/dense/Pow/yConst*
_output_shapes
: *
valueB
 *  @@*
dtype0
½
+bert/encoder/layer_7/intermediate/dense/PowPow/bert/encoder/layer_7/intermediate/dense/BiasAdd-bert/encoder/layer_7/intermediate/dense/Pow/y*
T0* 
_output_shapes
:

r
-bert/encoder/layer_7/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
¹
+bert/encoder/layer_7/intermediate/dense/mulMul-bert/encoder/layer_7/intermediate/dense/mul/x+bert/encoder/layer_7/intermediate/dense/Pow* 
_output_shapes
:
*
T0
»
+bert/encoder/layer_7/intermediate/dense/addAdd/bert/encoder/layer_7/intermediate/dense/BiasAdd+bert/encoder/layer_7/intermediate/dense/mul* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_7/intermediate/dense/mul_1/xConst*
dtype0*
_output_shapes
: *
valueB
 **BL?
½
-bert/encoder/layer_7/intermediate/dense/mul_1Mul/bert/encoder/layer_7/intermediate/dense/mul_1/x+bert/encoder/layer_7/intermediate/dense/add*
T0* 
_output_shapes
:


,bert/encoder/layer_7/intermediate/dense/TanhTanh-bert/encoder/layer_7/intermediate/dense/mul_1*
T0* 
_output_shapes
:

t
/bert/encoder/layer_7/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¾
-bert/encoder/layer_7/intermediate/dense/add_1Add/bert/encoder/layer_7/intermediate/dense/add_1/x,bert/encoder/layer_7/intermediate/dense/Tanh*
T0* 
_output_shapes
:

t
/bert/encoder/layer_7/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
¿
-bert/encoder/layer_7/intermediate/dense/mul_2Mul/bert/encoder/layer_7/intermediate/dense/mul_2/x-bert/encoder/layer_7/intermediate/dense/add_1*
T0* 
_output_shapes
:

¿
-bert/encoder/layer_7/intermediate/dense/mul_3Mul/bert/encoder/layer_7/intermediate/dense/BiasAdd-bert/encoder/layer_7/intermediate/dense/mul_2*
T0* 
_output_shapes
:

Ù
Kbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
dtype0*
_output_shapes
: 
Î
Lbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
seed2 
Ý
Ibert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel* 
_output_shapes
:

Ý
(bert/encoder/layer_7/output/dense/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
	container 
»
/bert/encoder/layer_7/output/dense/kernel/AssignAssign(bert/encoder/layer_7/output/dense/kernelEbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
validate_shape(* 
_output_shapes
:

Ë
-bert/encoder/layer_7/output/dense/kernel/readIdentity(bert/encoder/layer_7/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel* 
_output_shapes
:

Â
8bert/encoder/layer_7/output/dense/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
dtype0
Ï
&bert/encoder/layer_7/output/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias
£
-bert/encoder/layer_7/output/dense/bias/AssignAssign&bert/encoder/layer_7/output/dense/bias8bert/encoder/layer_7/output/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias
À
+bert/encoder/layer_7/output/dense/bias/readIdentity&bert/encoder/layer_7/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
_output_shapes	
:
á
(bert/encoder/layer_7/output/dense/MatMulMatMul-bert/encoder/layer_7/intermediate/dense/mul_3-bert/encoder/layer_7/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
Í
)bert/encoder/layer_7/output/dense/BiasAddBiasAdd(bert/encoder/layer_7/output/dense/MatMul+bert/encoder/layer_7/output/dense/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC
½
bert/encoder/layer_7/output/addAdd)bert/encoder/layer_7/output/dense/BiasAdd?bert/encoder/layer_7/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ê
<bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_7/output/LayerNorm/beta
VariableV2*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
³
1bert/encoder/layer_7/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_7/output/LayerNorm/beta<bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zeros*
T0*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
Ì
/bert/encoder/layer_7/output/LayerNorm/beta/readIdentity*bert/encoder/layer_7/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
_output_shapes	
:
Ë
<bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Ù
+bert/encoder/layer_7/output/LayerNorm/gamma
VariableV2*
shared_name *>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
¶
2bert/encoder/layer_7/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_7/output/LayerNorm/gamma<bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_7/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_7/output/LayerNorm/gamma*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
_output_shapes	
:*
T0

Dbert/encoder/layer_7/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
è
2bert/encoder/layer_7/output/LayerNorm/moments/meanMeanbert/encoder/layer_7/output/addDbert/encoder/layer_7/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
¨
:bert/encoder/layer_7/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_7/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_7/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_7/output/add:bert/encoder/layer_7/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
*
T0

Hbert/encoder/layer_7/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

6bert/encoder/layer_7/output/LayerNorm/moments/varianceMean?bert/encoder/layer_7/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_7/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
z
5bert/encoder/layer_7/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_7/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_7/output/LayerNorm/moments/variance5bert/encoder/layer_7/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	

5bert/encoder/layer_7/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_7/output/LayerNorm/batchnorm/add*
_output_shapes
:	*
T0
Î
3bert/encoder/layer_7/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_7/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_7/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

½
5bert/encoder/layer_7/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_7/output/add3bert/encoder/layer_7/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
5bert/encoder/layer_7/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_7/output/LayerNorm/moments/mean3bert/encoder/layer_7/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
Í
3bert/encoder/layer_7/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_7/output/LayerNorm/beta/read5bert/encoder/layer_7/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ó
5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_7/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_7/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
é
Sbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
dtype0
Û
]bert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ý
Qbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel
ë
Mbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel* 
_output_shapes
:

í
0bert/encoder/layer_8/attention/self/query/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel
Û
7bert/encoder/layer_8/attention/self/query/kernel/AssignAssign0bert/encoder/layer_8/attention/self/query/kernelMbert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_8/attention/self/query/kernel/readIdentity0bert/encoder/layer_8/attention/self/query/kernel* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel
Ò
@bert/encoder/layer_8/attention/self/query/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_8/attention/self/query/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
	container *
shape:
Ã
5bert/encoder/layer_8/attention/self/query/bias/AssignAssign.bert/encoder/layer_8/attention/self/query/bias@bert/encoder/layer_8/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_8/attention/self/query/bias/readIdentity.bert/encoder/layer_8/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
_output_shapes	
:
ù
0bert/encoder/layer_8/attention/self/query/MatMulMatMul5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_15bert/encoder/layer_8/attention/self/query/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
å
1bert/encoder/layer_8/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_8/attention/self/query/MatMul3bert/encoder/layer_8/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

å
Qbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
õ
Obert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel* 
_output_shapes
:

é
.bert/encoder/layer_8/attention/self/key/kernel
VariableV2*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
Ó
5bert/encoder/layer_8/attention/self/key/kernel/AssignAssign.bert/encoder/layer_8/attention/self/key/kernelKbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_8/attention/self/key/kernel/readIdentity.bert/encoder/layer_8/attention/self/key/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel
Î
>bert/encoder/layer_8/attention/self/key/bias/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
dtype0*
_output_shapes	
:
Û
,bert/encoder/layer_8/attention/self/key/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
	container 
»
3bert/encoder/layer_8/attention/self/key/bias/AssignAssign,bert/encoder/layer_8/attention/self/key/bias>bert/encoder/layer_8/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_8/attention/self/key/bias/readIdentity,bert/encoder/layer_8/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
_output_shapes	
:
õ
.bert/encoder/layer_8/attention/self/key/MatMulMatMul5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_13bert/encoder/layer_8/attention/self/key/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ß
/bert/encoder/layer_8/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_8/attention/self/key/MatMul1bert/encoder/layer_8/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

é
Sbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
seed2 *
dtype0
ý
Qbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel
ë
Mbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel* 
_output_shapes
:

í
0bert/encoder/layer_8/attention/self/value/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel
Û
7bert/encoder/layer_8/attention/self/value/kernel/AssignAssign0bert/encoder/layer_8/attention/self/value/kernelMbert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_8/attention/self/value/kernel/readIdentity0bert/encoder/layer_8/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_8/attention/self/value/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias
ß
.bert/encoder/layer_8/attention/self/value/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ã
5bert/encoder/layer_8/attention/self/value/bias/AssignAssign.bert/encoder/layer_8/attention/self/value/bias@bert/encoder/layer_8/attention/self/value/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_8/attention/self/value/bias/readIdentity.bert/encoder/layer_8/attention/self/value/bias*
_output_shapes	
:*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias
ù
0bert/encoder/layer_8/attention/self/value/MatMulMatMul5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_15bert/encoder/layer_8/attention/self/value/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
å
1bert/encoder/layer_8/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_8/attention/self/value/MatMul3bert/encoder/layer_8/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:


1bert/encoder/layer_8/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Ü
+bert/encoder/layer_8/attention/self/ReshapeReshape1bert/encoder/layer_8/attention/self/query/BiasAdd1bert/encoder/layer_8/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:@

2bert/encoder/layer_8/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_8/attention/self/transpose	Transpose+bert/encoder/layer_8/attention/self/Reshape2bert/encoder/layer_8/attention/self/transpose/perm*
T0*'
_output_shapes
:@*
Tperm0

3bert/encoder/layer_8/attention/self/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Þ
-bert/encoder/layer_8/attention/self/Reshape_1Reshape/bert/encoder/layer_8/attention/self/key/BiasAdd3bert/encoder/layer_8/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_8/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_8/attention/self/transpose_1	Transpose-bert/encoder/layer_8/attention/self/Reshape_14bert/encoder/layer_8/attention/self/transpose_1/perm*'
_output_shapes
:@*
Tperm0*
T0
è
*bert/encoder/layer_8/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_8/attention/self/transpose/bert/encoder/layer_8/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:
n
)bert/encoder/layer_8/attention/self/Mul/yConst*
_output_shapes
: *
valueB
 *   >*
dtype0
¸
'bert/encoder/layer_8/attention/self/MulMul*bert/encoder/layer_8/attention/self/MatMul)bert/encoder/layer_8/attention/self/Mul/y*
T0*(
_output_shapes
:
|
2bert/encoder/layer_8/attention/self/ExpandDims/dimConst*
dtype0*
_output_shapes
:*
valueB:
Á
.bert/encoder/layer_8/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_8/attention/self/ExpandDims/dim*

Tdim0*
T0*(
_output_shapes
:
n
)bert/encoder/layer_8/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¼
'bert/encoder/layer_8/attention/self/subSub)bert/encoder/layer_8/attention/self/sub/x.bert/encoder/layer_8/attention/self/ExpandDims*(
_output_shapes
:*
T0
p
+bert/encoder/layer_8/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¹
)bert/encoder/layer_8/attention/self/mul_1Mul'bert/encoder/layer_8/attention/self/sub+bert/encoder/layer_8/attention/self/mul_1/y*(
_output_shapes
:*
T0
µ
'bert/encoder/layer_8/attention/self/addAdd'bert/encoder/layer_8/attention/self/Mul)bert/encoder/layer_8/attention/self/mul_1*
T0*(
_output_shapes
:

+bert/encoder/layer_8/attention/self/SoftmaxSoftmax'bert/encoder/layer_8/attention/self/add*
T0*(
_output_shapes
:

3bert/encoder/layer_8/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
à
-bert/encoder/layer_8/attention/self/Reshape_2Reshape1bert/encoder/layer_8/attention/self/value/BiasAdd3bert/encoder/layer_8/attention/self/Reshape_2/shape*'
_output_shapes
:@*
T0*
Tshape0

4bert/encoder/layer_8/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_8/attention/self/transpose_2	Transpose-bert/encoder/layer_8/attention/self/Reshape_24bert/encoder/layer_8/attention/self/transpose_2/perm*'
_output_shapes
:@*
Tperm0*
T0
ç
,bert/encoder/layer_8/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_8/attention/self/Softmax/bert/encoder/layer_8/attention/self/transpose_2*'
_output_shapes
:@*
adj_x( *
adj_y( *
T0

4bert/encoder/layer_8/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ß
/bert/encoder/layer_8/attention/self/transpose_3	Transpose,bert/encoder/layer_8/attention/self/MatMul_14bert/encoder/layer_8/attention/self/transpose_3/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_8/attention/self/Reshape_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
×
-bert/encoder/layer_8/attention/self/Reshape_3Reshape/bert/encoder/layer_8/attention/self/transpose_33bert/encoder/layer_8/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:

í
Ubert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
dtype0*
_output_shapes
:
à
Tbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
â
Vbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
á
_bert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 

Sbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel
ó
Obert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal/mean*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel* 
_output_shapes
:
*
T0
ñ
2bert/encoder/layer_8/attention/output/dense/kernel
VariableV2* 
_output_shapes
:
*
shared_name *E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
	container *
shape:
*
dtype0
ã
9bert/encoder/layer_8/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_8/attention/output/dense/kernelObert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
é
7bert/encoder/layer_8/attention/output/dense/kernel/readIdentity2bert/encoder/layer_8/attention/output/dense/kernel* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel
Ö
Bbert/encoder/layer_8/attention/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias
ã
0bert/encoder/layer_8/attention/output/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
	container *
shape:
Ë
7bert/encoder/layer_8/attention/output/dense/bias/AssignAssign0bert/encoder/layer_8/attention/output/dense/biasBbert/encoder/layer_8/attention/output/dense/bias/Initializer/zeros*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
Þ
5bert/encoder/layer_8/attention/output/dense/bias/readIdentity0bert/encoder/layer_8/attention/output/dense/bias*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
_output_shapes	
:
õ
2bert/encoder/layer_8/attention/output/dense/MatMulMatMul-bert/encoder/layer_8/attention/self/Reshape_37bert/encoder/layer_8/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ë
3bert/encoder/layer_8/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_8/attention/output/dense/MatMul5bert/encoder/layer_8/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

Ç
)bert/encoder/layer_8/attention/output/addAdd3bert/encoder/layer_8/attention/output/dense/BiasAdd5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Þ
Fbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
dtype0
ë
4bert/encoder/layer_8/attention/output/LayerNorm/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
	container 
Û
;bert/encoder/layer_8/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_8/attention/output/LayerNorm/betaFbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta
ê
9bert/encoder/layer_8/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_8/attention/output/LayerNorm/beta*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
_output_shapes	
:*
T0
ß
Fbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_8/attention/output/LayerNorm/gamma
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
	container *
shape:
Þ
<bert/encoder/layer_8/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_8/attention/output/LayerNorm/gammaFbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/ones*
T0*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
í
:bert/encoder/layer_8/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_8/attention/output/LayerNorm/gamma*
_output_shapes	
:*
T0*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma

Nbert/encoder/layer_8/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

<bert/encoder/layer_8/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_8/attention/output/addNbert/encoder/layer_8/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¼
Dbert/encoder/layer_8/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_8/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ú
Ibert/encoder/layer_8/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_8/attention/output/addDbert/encoder/layer_8/attention/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
*
T0

Rbert/encoder/layer_8/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_8/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_8/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_8/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0

?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_8/attention/output/LayerNorm/moments/variance?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0
±
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ì
=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_8/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Û
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_8/attention/output/add=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_8/attention/output/LayerNorm/moments/mean=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

ë
=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_8/attention/output/LayerNorm/beta/read?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ñ
?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
å
Qbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Õ
[bert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:

õ
Obert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel* 
_output_shapes
:

é
.bert/encoder/layer_8/intermediate/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
	container *
shape:

Ó
5bert/encoder/layer_8/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_8/intermediate/dense/kernelKbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_8/intermediate/dense/kernel/readIdentity.bert/encoder/layer_8/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel* 
_output_shapes
:

Ú
Nbert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ê
Dbert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
dtype0*
_output_shapes
: 
Õ
>bert/encoder/layer_8/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias
Û
,bert/encoder/layer_8/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
	container *
shape:
»
3bert/encoder/layer_8/intermediate/dense/bias/AssignAssign,bert/encoder/layer_8/intermediate/dense/bias>bert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
validate_shape(
Ò
1bert/encoder/layer_8/intermediate/dense/bias/readIdentity,bert/encoder/layer_8/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
_output_shapes	
:
ÿ
.bert/encoder/layer_8/intermediate/dense/MatMulMatMul?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_8/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ß
/bert/encoder/layer_8/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_8/intermediate/dense/MatMul1bert/encoder/layer_8/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_8/intermediate/dense/Pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *  @@
½
+bert/encoder/layer_8/intermediate/dense/PowPow/bert/encoder/layer_8/intermediate/dense/BiasAdd-bert/encoder/layer_8/intermediate/dense/Pow/y* 
_output_shapes
:
*
T0
r
-bert/encoder/layer_8/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
¹
+bert/encoder/layer_8/intermediate/dense/mulMul-bert/encoder/layer_8/intermediate/dense/mul/x+bert/encoder/layer_8/intermediate/dense/Pow* 
_output_shapes
:
*
T0
»
+bert/encoder/layer_8/intermediate/dense/addAdd/bert/encoder/layer_8/intermediate/dense/BiasAdd+bert/encoder/layer_8/intermediate/dense/mul*
T0* 
_output_shapes
:

t
/bert/encoder/layer_8/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
½
-bert/encoder/layer_8/intermediate/dense/mul_1Mul/bert/encoder/layer_8/intermediate/dense/mul_1/x+bert/encoder/layer_8/intermediate/dense/add*
T0* 
_output_shapes
:


,bert/encoder/layer_8/intermediate/dense/TanhTanh-bert/encoder/layer_8/intermediate/dense/mul_1* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_8/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¾
-bert/encoder/layer_8/intermediate/dense/add_1Add/bert/encoder/layer_8/intermediate/dense/add_1/x,bert/encoder/layer_8/intermediate/dense/Tanh*
T0* 
_output_shapes
:

t
/bert/encoder/layer_8/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
¿
-bert/encoder/layer_8/intermediate/dense/mul_2Mul/bert/encoder/layer_8/intermediate/dense/mul_2/x-bert/encoder/layer_8/intermediate/dense/add_1* 
_output_shapes
:
*
T0
¿
-bert/encoder/layer_8/intermediate/dense/mul_3Mul/bert/encoder/layer_8/intermediate/dense/BiasAdd-bert/encoder/layer_8/intermediate/dense/mul_2*
T0* 
_output_shapes
:

Ù
Kbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel
Î
Lbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
dtype0
Ã
Ubert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel
Ý
Ibert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel
Ë
Ebert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel* 
_output_shapes
:

Ý
(bert/encoder/layer_8/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
	container *
shape:

»
/bert/encoder/layer_8/output/dense/kernel/AssignAssign(bert/encoder/layer_8/output/dense/kernelEbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ë
-bert/encoder/layer_8/output/dense/kernel/readIdentity(bert/encoder/layer_8/output/dense/kernel* 
_output_shapes
:
*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel
Â
8bert/encoder/layer_8/output/dense/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
dtype0
Ï
&bert/encoder/layer_8/output/dense/bias
VariableV2*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
£
-bert/encoder/layer_8/output/dense/bias/AssignAssign&bert/encoder/layer_8/output/dense/bias8bert/encoder/layer_8/output/dense/bias/Initializer/zeros*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
validate_shape(*
_output_shapes	
:
À
+bert/encoder/layer_8/output/dense/bias/readIdentity&bert/encoder/layer_8/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
_output_shapes	
:
á
(bert/encoder/layer_8/output/dense/MatMulMatMul-bert/encoder/layer_8/intermediate/dense/mul_3-bert/encoder/layer_8/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
Í
)bert/encoder/layer_8/output/dense/BiasAddBiasAdd(bert/encoder/layer_8/output/dense/MatMul+bert/encoder/layer_8/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

½
bert/encoder/layer_8/output/addAdd)bert/encoder/layer_8/output/dense/BiasAdd?bert/encoder/layer_8/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ê
<bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_8/output/LayerNorm/beta
VariableV2*
shared_name *=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
³
1bert/encoder/layer_8/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_8/output/LayerNorm/beta<bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zeros*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ì
/bert/encoder/layer_8/output/LayerNorm/beta/readIdentity*bert/encoder/layer_8/output/LayerNorm/beta*
T0*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
_output_shapes	
:
Ë
<bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Ù
+bert/encoder/layer_8/output/LayerNorm/gamma
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
	container 
¶
2bert/encoder/layer_8/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_8/output/LayerNorm/gamma<bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/ones*
T0*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Ï
0bert/encoder/layer_8/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_8/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
_output_shapes	
:

Dbert/encoder/layer_8/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
è
2bert/encoder/layer_8/output/LayerNorm/moments/meanMeanbert/encoder/layer_8/output/addDbert/encoder/layer_8/output/LayerNorm/moments/mean/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
¨
:bert/encoder/layer_8/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_8/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_8/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_8/output/add:bert/encoder/layer_8/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Hbert/encoder/layer_8/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

6bert/encoder/layer_8/output/LayerNorm/moments/varianceMean?bert/encoder/layer_8/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_8/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
z
5bert/encoder/layer_8/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_8/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_8/output/LayerNorm/moments/variance5bert/encoder/layer_8/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	

5bert/encoder/layer_8/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_8/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
Î
3bert/encoder/layer_8/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_8/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_8/output/LayerNorm/gamma/read* 
_output_shapes
:
*
T0
½
5bert/encoder/layer_8/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_8/output/add3bert/encoder/layer_8/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
5bert/encoder/layer_8/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_8/output/LayerNorm/moments/mean3bert/encoder/layer_8/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Í
3bert/encoder/layer_8/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_8/output/LayerNorm/beta/read5bert/encoder/layer_8/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
*
T0
Ó
5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_8/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_8/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
é
Sbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Û
]bert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/shape*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:

ý
Qbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel
ë
Mbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel* 
_output_shapes
:

í
0bert/encoder/layer_9/attention/self/query/kernel
VariableV2*
shared_name *C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Û
7bert/encoder/layer_9/attention/self/query/kernel/AssignAssign0bert/encoder/layer_9/attention/self/query/kernelMbert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_9/attention/self/query/kernel/readIdentity0bert/encoder/layer_9/attention/self/query/kernel*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel* 
_output_shapes
:
*
T0
Ò
@bert/encoder/layer_9/attention/self/query/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_9/attention/self/query/bias
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ã
5bert/encoder/layer_9/attention/self/query/bias/AssignAssign.bert/encoder/layer_9/attention/self/query/bias@bert/encoder/layer_9/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_9/attention/self/query/bias/readIdentity.bert/encoder/layer_9/attention/self/query/bias*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
_output_shapes	
:
ù
0bert/encoder/layer_9/attention/self/query/MatMulMatMul5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_15bert/encoder/layer_9/attention/self/query/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_9/attention/self/query/BiasAddBiasAdd0bert/encoder/layer_9/attention/self/query/MatMul3bert/encoder/layer_9/attention/self/query/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
å
Qbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ú
Rbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel
Õ
[bert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
seed2 
õ
Obert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel* 
_output_shapes
:

é
.bert/encoder/layer_9/attention/self/key/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel
Ó
5bert/encoder/layer_9/attention/self/key/kernel/AssignAssign.bert/encoder/layer_9/attention/self/key/kernelKbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_9/attention/self/key/kernel/readIdentity.bert/encoder/layer_9/attention/self/key/kernel* 
_output_shapes
:
*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel
Î
>bert/encoder/layer_9/attention/self/key/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias
Û
,bert/encoder/layer_9/attention/self/key/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
	container *
shape:
»
3bert/encoder/layer_9/attention/self/key/bias/AssignAssign,bert/encoder/layer_9/attention/self/key/bias>bert/encoder/layer_9/attention/self/key/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
validate_shape(
Ò
1bert/encoder/layer_9/attention/self/key/bias/readIdentity,bert/encoder/layer_9/attention/self/key/bias*
T0*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
_output_shapes	
:
õ
.bert/encoder/layer_9/attention/self/key/MatMulMatMul5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_13bert/encoder/layer_9/attention/self/key/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
ß
/bert/encoder/layer_9/attention/self/key/BiasAddBiasAdd.bert/encoder/layer_9/attention/self/key/MatMul1bert/encoder/layer_9/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

é
Sbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
dtype0*
_output_shapes
:
Ü
Rbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
Tbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
dtype0
Û
]bert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel
ý
Qbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel* 
_output_shapes
:

í
0bert/encoder/layer_9/attention/self/value/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
	container *
shape:

Û
7bert/encoder/layer_9/attention/self/value/kernel/AssignAssign0bert/encoder/layer_9/attention/self/value/kernelMbert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ã
5bert/encoder/layer_9/attention/self/value/kernel/readIdentity0bert/encoder/layer_9/attention/self/value/kernel*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel* 
_output_shapes
:

Ò
@bert/encoder/layer_9/attention/self/value/bias/Initializer/zerosConst*
valueB*    *A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
dtype0*
_output_shapes	
:
ß
.bert/encoder/layer_9/attention/self/value/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias
Ã
5bert/encoder/layer_9/attention/self/value/bias/AssignAssign.bert/encoder/layer_9/attention/self/value/bias@bert/encoder/layer_9/attention/self/value/bias/Initializer/zeros*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
Ø
3bert/encoder/layer_9/attention/self/value/bias/readIdentity.bert/encoder/layer_9/attention/self/value/bias*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
_output_shapes	
:*
T0
ù
0bert/encoder/layer_9/attention/self/value/MatMulMatMul5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_15bert/encoder/layer_9/attention/self/value/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
å
1bert/encoder/layer_9/attention/self/value/BiasAddBiasAdd0bert/encoder/layer_9/attention/self/value/MatMul3bert/encoder/layer_9/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:


1bert/encoder/layer_9/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
Ü
+bert/encoder/layer_9/attention/self/ReshapeReshape1bert/encoder/layer_9/attention/self/query/BiasAdd1bert/encoder/layer_9/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:@

2bert/encoder/layer_9/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ú
-bert/encoder/layer_9/attention/self/transpose	Transpose+bert/encoder/layer_9/attention/self/Reshape2bert/encoder/layer_9/attention/self/transpose/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_9/attention/self/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*%
valueB"         @   
Þ
-bert/encoder/layer_9/attention/self/Reshape_1Reshape/bert/encoder/layer_9/attention/self/key/BiasAdd3bert/encoder/layer_9/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_9/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_9/attention/self/transpose_1	Transpose-bert/encoder/layer_9/attention/self/Reshape_14bert/encoder/layer_9/attention/self/transpose_1/perm*'
_output_shapes
:@*
Tperm0*
T0
è
*bert/encoder/layer_9/attention/self/MatMulBatchMatMulV2-bert/encoder/layer_9/attention/self/transpose/bert/encoder/layer_9/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:
n
)bert/encoder/layer_9/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
¸
'bert/encoder/layer_9/attention/self/MulMul*bert/encoder/layer_9/attention/self/MatMul)bert/encoder/layer_9/attention/self/Mul/y*
T0*(
_output_shapes
:
|
2bert/encoder/layer_9/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Á
.bert/encoder/layer_9/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_9/attention/self/ExpandDims/dim*(
_output_shapes
:*

Tdim0*
T0
n
)bert/encoder/layer_9/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¼
'bert/encoder/layer_9/attention/self/subSub)bert/encoder/layer_9/attention/self/sub/x.bert/encoder/layer_9/attention/self/ExpandDims*
T0*(
_output_shapes
:
p
+bert/encoder/layer_9/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¹
)bert/encoder/layer_9/attention/self/mul_1Mul'bert/encoder/layer_9/attention/self/sub+bert/encoder/layer_9/attention/self/mul_1/y*(
_output_shapes
:*
T0
µ
'bert/encoder/layer_9/attention/self/addAdd'bert/encoder/layer_9/attention/self/Mul)bert/encoder/layer_9/attention/self/mul_1*(
_output_shapes
:*
T0

+bert/encoder/layer_9/attention/self/SoftmaxSoftmax'bert/encoder/layer_9/attention/self/add*(
_output_shapes
:*
T0

3bert/encoder/layer_9/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
à
-bert/encoder/layer_9/attention/self/Reshape_2Reshape1bert/encoder/layer_9/attention/self/value/BiasAdd3bert/encoder/layer_9/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:@

4bert/encoder/layer_9/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
à
/bert/encoder/layer_9/attention/self/transpose_2	Transpose-bert/encoder/layer_9/attention/self/Reshape_24bert/encoder/layer_9/attention/self/transpose_2/perm*'
_output_shapes
:@*
Tperm0*
T0
ç
,bert/encoder/layer_9/attention/self/MatMul_1BatchMatMulV2+bert/encoder/layer_9/attention/self/Softmax/bert/encoder/layer_9/attention/self/transpose_2*
T0*'
_output_shapes
:@*
adj_x( *
adj_y( 

4bert/encoder/layer_9/attention/self/transpose_3/permConst*
dtype0*
_output_shapes
:*%
valueB"             
ß
/bert/encoder/layer_9/attention/self/transpose_3	Transpose,bert/encoder/layer_9/attention/self/MatMul_14bert/encoder/layer_9/attention/self/transpose_3/perm*'
_output_shapes
:@*
Tperm0*
T0

3bert/encoder/layer_9/attention/self/Reshape_3/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
×
-bert/encoder/layer_9/attention/self/Reshape_3Reshape/bert/encoder/layer_9/attention/self/transpose_33bert/encoder/layer_9/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:

í
Ubert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel
à
Tbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
â
Vbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
á
_bert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 

Sbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel
ñ
2bert/encoder/layer_9/attention/output/dense/kernel
VariableV2*
shared_name *E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ã
9bert/encoder/layer_9/attention/output/dense/kernel/AssignAssign2bert/encoder/layer_9/attention/output/dense/kernelObert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
é
7bert/encoder/layer_9/attention/output/dense/kernel/readIdentity2bert/encoder/layer_9/attention/output/dense/kernel*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel* 
_output_shapes
:

Ö
Bbert/encoder/layer_9/attention/output/dense/bias/Initializer/zerosConst*
valueB*    *C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
dtype0*
_output_shapes	
:
ã
0bert/encoder/layer_9/attention/output/dense/bias
VariableV2*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ë
7bert/encoder/layer_9/attention/output/dense/bias/AssignAssign0bert/encoder/layer_9/attention/output/dense/biasBbert/encoder/layer_9/attention/output/dense/bias/Initializer/zeros*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Þ
5bert/encoder/layer_9/attention/output/dense/bias/readIdentity0bert/encoder/layer_9/attention/output/dense/bias*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
_output_shapes	
:*
T0
õ
2bert/encoder/layer_9/attention/output/dense/MatMulMatMul-bert/encoder/layer_9/attention/self/Reshape_37bert/encoder/layer_9/attention/output/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ë
3bert/encoder/layer_9/attention/output/dense/BiasAddBiasAdd2bert/encoder/layer_9/attention/output/dense/MatMul5bert/encoder/layer_9/attention/output/dense/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC
Ç
)bert/encoder/layer_9/attention/output/addAdd3bert/encoder/layer_9/attention/output/dense/BiasAdd5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Þ
Fbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta
ë
4bert/encoder/layer_9/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta*
	container *
shape:
Û
;bert/encoder/layer_9/attention/output/LayerNorm/beta/AssignAssign4bert/encoder/layer_9/attention/output/LayerNorm/betaFbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ê
9bert/encoder/layer_9/attention/output/LayerNorm/beta/readIdentity4bert/encoder/layer_9/attention/output/LayerNorm/beta*
_output_shapes	
:*
T0*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta
ß
Fbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_9/attention/output/LayerNorm/gamma
VariableV2*
shared_name *H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
Þ
<bert/encoder/layer_9/attention/output/LayerNorm/gamma/AssignAssign5bert/encoder/layer_9/attention/output/LayerNorm/gammaFbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/ones*
T0*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
í
:bert/encoder/layer_9/attention/output/LayerNorm/gamma/readIdentity5bert/encoder/layer_9/attention/output/LayerNorm/gamma*
T0*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
_output_shapes	
:

Nbert/encoder/layer_9/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:

<bert/encoder/layer_9/attention/output/LayerNorm/moments/meanMean)bert/encoder/layer_9/attention/output/addNbert/encoder/layer_9/attention/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
¼
Dbert/encoder/layer_9/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_9/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ú
Ibert/encoder/layer_9/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference)bert/encoder/layer_9/attention/output/addDbert/encoder/layer_9/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Rbert/encoder/layer_9/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
®
@bert/encoder/layer_9/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_9/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_9/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0

?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ñ
=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/addAdd@bert/encoder/layer_9/attention/output/LayerNorm/moments/variance?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add/y*
_output_shapes
:	*
T0
±
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add*
_output_shapes
:	*
T0
ì
=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/Rsqrt:bert/encoder/layer_9/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Û
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul_1Mul)bert/encoder/layer_9/attention/output/add=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_9/attention/output/LayerNorm/moments/mean=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
ë
=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/subSub9bert/encoder/layer_9/attention/output/LayerNorm/beta/read?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ñ
?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add_1Add?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
å
Qbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ø
Pbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel
Ú
Rbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
×£<*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel
Õ
[bert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
seed2 
õ
Obert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel* 
_output_shapes
:

ã
Kbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel* 
_output_shapes
:

é
.bert/encoder/layer_9/intermediate/dense/kernel
VariableV2*
shared_name *A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

Ó
5bert/encoder/layer_9/intermediate/dense/kernel/AssignAssign.bert/encoder/layer_9/intermediate/dense/kernelKbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

Ý
3bert/encoder/layer_9/intermediate/dense/kernel/readIdentity.bert/encoder/layer_9/intermediate/dense/kernel*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel* 
_output_shapes
:

Ú
Nbert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ê
Dbert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias
Õ
>bert/encoder/layer_9/intermediate/dense/bias/Initializer/zerosFillNbert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros/shape_as_tensorDbert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros/Const*
T0*

index_type0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
_output_shapes	
:
Û
,bert/encoder/layer_9/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
	container *
shape:
»
3bert/encoder/layer_9/intermediate/dense/bias/AssignAssign,bert/encoder/layer_9/intermediate/dense/bias>bert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_9/intermediate/dense/bias/readIdentity,bert/encoder/layer_9/intermediate/dense/bias*
T0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
_output_shapes	
:
ÿ
.bert/encoder/layer_9/intermediate/dense/MatMulMatMul?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add_13bert/encoder/layer_9/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
ß
/bert/encoder/layer_9/intermediate/dense/BiasAddBiasAdd.bert/encoder/layer_9/intermediate/dense/MatMul1bert/encoder/layer_9/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

r
-bert/encoder/layer_9/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
½
+bert/encoder/layer_9/intermediate/dense/PowPow/bert/encoder/layer_9/intermediate/dense/BiasAdd-bert/encoder/layer_9/intermediate/dense/Pow/y*
T0* 
_output_shapes
:

r
-bert/encoder/layer_9/intermediate/dense/mul/xConst*
_output_shapes
: *
valueB
 *'7=*
dtype0
¹
+bert/encoder/layer_9/intermediate/dense/mulMul-bert/encoder/layer_9/intermediate/dense/mul/x+bert/encoder/layer_9/intermediate/dense/Pow*
T0* 
_output_shapes
:

»
+bert/encoder/layer_9/intermediate/dense/addAdd/bert/encoder/layer_9/intermediate/dense/BiasAdd+bert/encoder/layer_9/intermediate/dense/mul*
T0* 
_output_shapes
:

t
/bert/encoder/layer_9/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
½
-bert/encoder/layer_9/intermediate/dense/mul_1Mul/bert/encoder/layer_9/intermediate/dense/mul_1/x+bert/encoder/layer_9/intermediate/dense/add*
T0* 
_output_shapes
:


,bert/encoder/layer_9/intermediate/dense/TanhTanh-bert/encoder/layer_9/intermediate/dense/mul_1* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_9/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¾
-bert/encoder/layer_9/intermediate/dense/add_1Add/bert/encoder/layer_9/intermediate/dense/add_1/x,bert/encoder/layer_9/intermediate/dense/Tanh* 
_output_shapes
:
*
T0
t
/bert/encoder/layer_9/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
¿
-bert/encoder/layer_9/intermediate/dense/mul_2Mul/bert/encoder/layer_9/intermediate/dense/mul_2/x-bert/encoder/layer_9/intermediate/dense/add_1*
T0* 
_output_shapes
:

¿
-bert/encoder/layer_9/intermediate/dense/mul_3Mul/bert/encoder/layer_9/intermediate/dense/BiasAdd-bert/encoder/layer_9/intermediate/dense/mul_2* 
_output_shapes
:
*
T0
Ù
Kbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
dtype0*
_output_shapes
:
Ì
Jbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
dtype0*
_output_shapes
: 
Î
Lbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
dtype0*
_output_shapes
: 
Ã
Ubert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/shape*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
Ý
Ibert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel* 
_output_shapes
:

Ý
(bert/encoder/layer_9/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
	container *
shape:

»
/bert/encoder/layer_9/output/dense/kernel/AssignAssign(bert/encoder/layer_9/output/dense/kernelEbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
validate_shape(* 
_output_shapes
:

Ë
-bert/encoder/layer_9/output/dense/kernel/readIdentity(bert/encoder/layer_9/output/dense/kernel*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel* 
_output_shapes
:

Â
8bert/encoder/layer_9/output/dense/bias/Initializer/zerosConst*
valueB*    *9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
dtype0*
_output_shapes	
:
Ï
&bert/encoder/layer_9/output/dense/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias
£
-bert/encoder/layer_9/output/dense/bias/AssignAssign&bert/encoder/layer_9/output/dense/bias8bert/encoder/layer_9/output/dense/bias/Initializer/zeros*
T0*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
À
+bert/encoder/layer_9/output/dense/bias/readIdentity&bert/encoder/layer_9/output/dense/bias*
T0*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
_output_shapes	
:
á
(bert/encoder/layer_9/output/dense/MatMulMatMul-bert/encoder/layer_9/intermediate/dense/mul_3-bert/encoder/layer_9/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
Í
)bert/encoder/layer_9/output/dense/BiasAddBiasAdd(bert/encoder/layer_9/output/dense/MatMul+bert/encoder/layer_9/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

½
bert/encoder/layer_9/output/addAdd)bert/encoder/layer_9/output/dense/BiasAdd?bert/encoder/layer_9/attention/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
*
T0
Ê
<bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
×
*bert/encoder/layer_9/output/LayerNorm/beta
VariableV2*
shared_name *=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
	container *
shape:*
dtype0*
_output_shapes	
:
³
1bert/encoder/layer_9/output/LayerNorm/beta/AssignAssign*bert/encoder/layer_9/output/LayerNorm/beta<bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta
Ì
/bert/encoder/layer_9/output/LayerNorm/beta/readIdentity*bert/encoder/layer_9/output/LayerNorm/beta*
_output_shapes	
:*
T0*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta
Ë
<bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Ù
+bert/encoder/layer_9/output/LayerNorm/gamma
VariableV2*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
¶
2bert/encoder/layer_9/output/LayerNorm/gamma/AssignAssign+bert/encoder/layer_9/output/LayerNorm/gamma<bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_9/output/LayerNorm/gamma/readIdentity+bert/encoder/layer_9/output/LayerNorm/gamma*
T0*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
_output_shapes	
:

Dbert/encoder/layer_9/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
è
2bert/encoder/layer_9/output/LayerNorm/moments/meanMeanbert/encoder/layer_9/output/addDbert/encoder/layer_9/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
¨
:bert/encoder/layer_9/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_9/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
Ü
?bert/encoder/layer_9/output/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/encoder/layer_9/output/add:bert/encoder/layer_9/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Hbert/encoder/layer_9/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0

6bert/encoder/layer_9/output/LayerNorm/moments/varianceMean?bert/encoder/layer_9/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_9/output/LayerNorm/moments/variance/reduction_indices*
T0*
_output_shapes
:	*
	keep_dims(*

Tidx0
z
5bert/encoder/layer_9/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ó
3bert/encoder/layer_9/output/LayerNorm/batchnorm/addAdd6bert/encoder/layer_9/output/LayerNorm/moments/variance5bert/encoder/layer_9/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	

5bert/encoder/layer_9/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_9/output/LayerNorm/batchnorm/add*
_output_shapes
:	*
T0
Î
3bert/encoder/layer_9/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_9/output/LayerNorm/batchnorm/Rsqrt0bert/encoder/layer_9/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

½
5bert/encoder/layer_9/output/LayerNorm/batchnorm/mul_1Mulbert/encoder/layer_9/output/add3bert/encoder/layer_9/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
5bert/encoder/layer_9/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_9/output/LayerNorm/moments/mean3bert/encoder/layer_9/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Í
3bert/encoder/layer_9/output/LayerNorm/batchnorm/subSub/bert/encoder/layer_9/output/LayerNorm/beta/read5bert/encoder/layer_9/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ó
5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_1Add5bert/encoder/layer_9/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_9/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

ë
Tbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
dtype0*
_output_shapes
:
Þ
Sbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
dtype0*
_output_shapes
: 
à
Ubert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
^bert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 

Rbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/mulMul^bert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalUbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel* 
_output_shapes
:

ï
Nbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normalAddRbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/mulSbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel
ï
1bert/encoder/layer_10/attention/self/query/kernel
VariableV2*
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
	container 
ß
8bert/encoder/layer_10/attention/self/query/kernel/AssignAssign1bert/encoder/layer_10/attention/self/query/kernelNbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
æ
6bert/encoder/layer_10/attention/self/query/kernel/readIdentity1bert/encoder/layer_10/attention/self/query/kernel*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel* 
_output_shapes
:

Ô
Abert/encoder/layer_10/attention/self/query/bias/Initializer/zerosConst*
valueB*    *B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
dtype0*
_output_shapes	
:
á
/bert/encoder/layer_10/attention/self/query/bias
VariableV2*
shared_name *B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ç
6bert/encoder/layer_10/attention/self/query/bias/AssignAssign/bert/encoder/layer_10/attention/self/query/biasAbert/encoder/layer_10/attention/self/query/bias/Initializer/zeros*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
Û
4bert/encoder/layer_10/attention/self/query/bias/readIdentity/bert/encoder/layer_10/attention/self/query/bias*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
_output_shapes	
:
û
1bert/encoder/layer_10/attention/self/query/MatMulMatMul5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_16bert/encoder/layer_10/attention/self/query/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
è
2bert/encoder/layer_10/attention/self/query/BiasAddBiasAdd1bert/encoder/layer_10/attention/self/query/MatMul4bert/encoder/layer_10/attention/self/query/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0
ç
Rbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ú
Qbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
dtype0*
_output_shapes
: 
Ü
Sbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
dtype0
Ø
\bert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/shape*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ù
Pbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/mulMul\bert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel* 
_output_shapes
:

ç
Lbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normalAddPbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/mulQbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel
ë
/bert/encoder/layer_10/attention/self/key/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
	container *
shape:

×
6bert/encoder/layer_10/attention/self/key/kernel/AssignAssign/bert/encoder/layer_10/attention/self/key/kernelLbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
validate_shape(
à
4bert/encoder/layer_10/attention/self/key/kernel/readIdentity/bert/encoder/layer_10/attention/self/key/kernel* 
_output_shapes
:
*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel
Ð
?bert/encoder/layer_10/attention/self/key/bias/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
dtype0*
_output_shapes	
:
Ý
-bert/encoder/layer_10/attention/self/key/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
	container 
¿
4bert/encoder/layer_10/attention/self/key/bias/AssignAssign-bert/encoder/layer_10/attention/self/key/bias?bert/encoder/layer_10/attention/self/key/bias/Initializer/zeros*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Õ
2bert/encoder/layer_10/attention/self/key/bias/readIdentity-bert/encoder/layer_10/attention/self/key/bias*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
_output_shapes	
:*
T0
÷
/bert/encoder/layer_10/attention/self/key/MatMulMatMul5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_14bert/encoder/layer_10/attention/self/key/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
â
0bert/encoder/layer_10/attention/self/key/BiasAddBiasAdd/bert/encoder/layer_10/attention/self/key/MatMul2bert/encoder/layer_10/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

ë
Tbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
dtype0*
_output_shapes
:
Þ
Sbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
dtype0*
_output_shapes
: 
à
Ubert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
^bert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
seed2 

Rbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/mulMul^bert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalUbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel* 
_output_shapes
:

ï
Nbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normalAddRbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/mulSbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel* 
_output_shapes
:

ï
1bert/encoder/layer_10/attention/self/value/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
	container *
shape:

ß
8bert/encoder/layer_10/attention/self/value/kernel/AssignAssign1bert/encoder/layer_10/attention/self/value/kernelNbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

æ
6bert/encoder/layer_10/attention/self/value/kernel/readIdentity1bert/encoder/layer_10/attention/self/value/kernel*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel* 
_output_shapes
:

Ô
Abert/encoder/layer_10/attention/self/value/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias
á
/bert/encoder/layer_10/attention/self/value/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias
Ç
6bert/encoder/layer_10/attention/self/value/bias/AssignAssign/bert/encoder/layer_10/attention/self/value/biasAbert/encoder/layer_10/attention/self/value/bias/Initializer/zeros*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Û
4bert/encoder/layer_10/attention/self/value/bias/readIdentity/bert/encoder/layer_10/attention/self/value/bias*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias*
_output_shapes	
:*
T0
û
1bert/encoder/layer_10/attention/self/value/MatMulMatMul5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_16bert/encoder/layer_10/attention/self/value/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
è
2bert/encoder/layer_10/attention/self/value/BiasAddBiasAdd1bert/encoder/layer_10/attention/self/value/MatMul4bert/encoder/layer_10/attention/self/value/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:


2bert/encoder/layer_10/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ß
,bert/encoder/layer_10/attention/self/ReshapeReshape2bert/encoder/layer_10/attention/self/query/BiasAdd2bert/encoder/layer_10/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:@

3bert/encoder/layer_10/attention/self/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
Ý
.bert/encoder/layer_10/attention/self/transpose	Transpose,bert/encoder/layer_10/attention/self/Reshape3bert/encoder/layer_10/attention/self/transpose/perm*
T0*'
_output_shapes
:@*
Tperm0

4bert/encoder/layer_10/attention/self/Reshape_1/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
á
.bert/encoder/layer_10/attention/self/Reshape_1Reshape0bert/encoder/layer_10/attention/self/key/BiasAdd4bert/encoder/layer_10/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:@

5bert/encoder/layer_10/attention/self/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ã
0bert/encoder/layer_10/attention/self/transpose_1	Transpose.bert/encoder/layer_10/attention/self/Reshape_15bert/encoder/layer_10/attention/self/transpose_1/perm*'
_output_shapes
:@*
Tperm0*
T0
ë
+bert/encoder/layer_10/attention/self/MatMulBatchMatMulV2.bert/encoder/layer_10/attention/self/transpose0bert/encoder/layer_10/attention/self/transpose_1*(
_output_shapes
:*
adj_x( *
adj_y(*
T0
o
*bert/encoder/layer_10/attention/self/Mul/yConst*
_output_shapes
: *
valueB
 *   >*
dtype0
»
(bert/encoder/layer_10/attention/self/MulMul+bert/encoder/layer_10/attention/self/MatMul*bert/encoder/layer_10/attention/self/Mul/y*
T0*(
_output_shapes
:
}
3bert/encoder/layer_10/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Ã
/bert/encoder/layer_10/attention/self/ExpandDims
ExpandDimsbert/encoder/mul3bert/encoder/layer_10/attention/self/ExpandDims/dim*(
_output_shapes
:*

Tdim0*
T0
o
*bert/encoder/layer_10/attention/self/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
¿
(bert/encoder/layer_10/attention/self/subSub*bert/encoder/layer_10/attention/self/sub/x/bert/encoder/layer_10/attention/self/ExpandDims*(
_output_shapes
:*
T0
q
,bert/encoder/layer_10/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¼
*bert/encoder/layer_10/attention/self/mul_1Mul(bert/encoder/layer_10/attention/self/sub,bert/encoder/layer_10/attention/self/mul_1/y*
T0*(
_output_shapes
:
¸
(bert/encoder/layer_10/attention/self/addAdd(bert/encoder/layer_10/attention/self/Mul*bert/encoder/layer_10/attention/self/mul_1*(
_output_shapes
:*
T0

,bert/encoder/layer_10/attention/self/SoftmaxSoftmax(bert/encoder/layer_10/attention/self/add*
T0*(
_output_shapes
:

4bert/encoder/layer_10/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ã
.bert/encoder/layer_10/attention/self/Reshape_2Reshape2bert/encoder/layer_10/attention/self/value/BiasAdd4bert/encoder/layer_10/attention/self/Reshape_2/shape*'
_output_shapes
:@*
T0*
Tshape0

5bert/encoder/layer_10/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ã
0bert/encoder/layer_10/attention/self/transpose_2	Transpose.bert/encoder/layer_10/attention/self/Reshape_25bert/encoder/layer_10/attention/self/transpose_2/perm*'
_output_shapes
:@*
Tperm0*
T0
ê
-bert/encoder/layer_10/attention/self/MatMul_1BatchMatMulV2,bert/encoder/layer_10/attention/self/Softmax0bert/encoder/layer_10/attention/self/transpose_2*
adj_x( *
adj_y( *
T0*'
_output_shapes
:@

5bert/encoder/layer_10/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
â
0bert/encoder/layer_10/attention/self/transpose_3	Transpose-bert/encoder/layer_10/attention/self/MatMul_15bert/encoder/layer_10/attention/self/transpose_3/perm*
T0*'
_output_shapes
:@*
Tperm0

4bert/encoder/layer_10/attention/self/Reshape_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ú
.bert/encoder/layer_10/attention/self/Reshape_3Reshape0bert/encoder/layer_10/attention/self/transpose_34bert/encoder/layer_10/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:

ï
Vbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
_output_shapes
:*
valueB"      *F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
dtype0
â
Ubert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
ä
Wbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
ä
`bert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalVbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
seed2 

Tbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/mulMul`bert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalWbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel
÷
Pbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normalAddTbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/mulUbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel* 
_output_shapes
:

ó
3bert/encoder/layer_10/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
	container *
shape:

ç
:bert/encoder/layer_10/attention/output/dense/kernel/AssignAssign3bert/encoder/layer_10/attention/output/dense/kernelPbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ì
8bert/encoder/layer_10/attention/output/dense/kernel/readIdentity3bert/encoder/layer_10/attention/output/dense/kernel* 
_output_shapes
:
*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel
Ø
Cbert/encoder/layer_10/attention/output/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias
å
1bert/encoder/layer_10/attention/output/dense/bias
VariableV2*
shared_name *D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ï
8bert/encoder/layer_10/attention/output/dense/bias/AssignAssign1bert/encoder/layer_10/attention/output/dense/biasCbert/encoder/layer_10/attention/output/dense/bias/Initializer/zeros*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
á
6bert/encoder/layer_10/attention/output/dense/bias/readIdentity1bert/encoder/layer_10/attention/output/dense/bias*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
_output_shapes	
:
ø
3bert/encoder/layer_10/attention/output/dense/MatMulMatMul.bert/encoder/layer_10/attention/self/Reshape_38bert/encoder/layer_10/attention/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
î
4bert/encoder/layer_10/attention/output/dense/BiasAddBiasAdd3bert/encoder/layer_10/attention/output/dense/MatMul6bert/encoder/layer_10/attention/output/dense/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC
É
*bert/encoder/layer_10/attention/output/addAdd4bert/encoder/layer_10/attention/output/dense/BiasAdd5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
*
T0
à
Gbert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_10/attention/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
	container *
shape:
ß
<bert/encoder/layer_10/attention/output/LayerNorm/beta/AssignAssign5bert/encoder/layer_10/attention/output/LayerNorm/betaGbert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zeros*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
í
:bert/encoder/layer_10/attention/output/LayerNorm/beta/readIdentity5bert/encoder/layer_10/attention/output/LayerNorm/beta*
T0*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
_output_shapes	
:
á
Gbert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
ï
6bert/encoder/layer_10/attention/output/LayerNorm/gamma
VariableV2*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
â
=bert/encoder/layer_10/attention/output/LayerNorm/gamma/AssignAssign6bert/encoder/layer_10/attention/output/LayerNorm/gammaGbert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/ones*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ð
;bert/encoder/layer_10/attention/output/LayerNorm/gamma/readIdentity6bert/encoder/layer_10/attention/output/LayerNorm/gamma*
T0*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
_output_shapes	
:

Obert/encoder/layer_10/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0

=bert/encoder/layer_10/attention/output/LayerNorm/moments/meanMean*bert/encoder/layer_10/attention/output/addObert/encoder/layer_10/attention/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
¾
Ebert/encoder/layer_10/attention/output/LayerNorm/moments/StopGradientStopGradient=bert/encoder/layer_10/attention/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ý
Jbert/encoder/layer_10/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference*bert/encoder/layer_10/attention/output/addEbert/encoder/layer_10/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Sbert/encoder/layer_10/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
±
Abert/encoder/layer_10/attention/output/LayerNorm/moments/varianceMeanJbert/encoder/layer_10/attention/output/LayerNorm/moments/SquaredDifferenceSbert/encoder/layer_10/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0

@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ô
>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/addAddAbert/encoder/layer_10/attention/output/LayerNorm/moments/variance@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	
³
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/RsqrtRsqrt>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ï
>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mulMul@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/Rsqrt;bert/encoder/layer_10/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Þ
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul_1Mul*bert/encoder/layer_10/attention/output/add>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

ñ
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul_2Mul=bert/encoder/layer_10/attention/output/LayerNorm/moments/mean>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

î
>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/subSub:bert/encoder/layer_10/attention/output/LayerNorm/beta/read@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul_2* 
_output_shapes
:
*
T0
ô
@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add_1Add@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/mul_1>bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
ç
Rbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ú
Qbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ü
Sbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ø
\bert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/shape* 
_output_shapes
:
*

seed *
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
seed2 *
dtype0
ù
Pbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/mulMul\bert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel
ç
Lbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normalAddPbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/mulQbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel* 
_output_shapes
:

ë
/bert/encoder/layer_10/intermediate/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
	container *
shape:

×
6bert/encoder/layer_10/intermediate/dense/kernel/AssignAssign/bert/encoder/layer_10/intermediate/dense/kernelLbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
à
4bert/encoder/layer_10/intermediate/dense/kernel/readIdentity/bert/encoder/layer_10/intermediate/dense/kernel* 
_output_shapes
:
*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel
Ü
Obert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ì
Ebert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias
Ù
?bert/encoder/layer_10/intermediate/dense/bias/Initializer/zerosFillObert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros/shape_as_tensorEbert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros/Const*
_output_shapes	
:*
T0*

index_type0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias
Ý
-bert/encoder/layer_10/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
	container *
shape:
¿
4bert/encoder/layer_10/intermediate/dense/bias/AssignAssign-bert/encoder/layer_10/intermediate/dense/bias?bert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias
Õ
2bert/encoder/layer_10/intermediate/dense/bias/readIdentity-bert/encoder/layer_10/intermediate/dense/bias*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
_output_shapes	
:

/bert/encoder/layer_10/intermediate/dense/MatMulMatMul@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add_14bert/encoder/layer_10/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
â
0bert/encoder/layer_10/intermediate/dense/BiasAddBiasAdd/bert/encoder/layer_10/intermediate/dense/MatMul2bert/encoder/layer_10/intermediate/dense/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC
s
.bert/encoder/layer_10/intermediate/dense/Pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *  @@
À
,bert/encoder/layer_10/intermediate/dense/PowPow0bert/encoder/layer_10/intermediate/dense/BiasAdd.bert/encoder/layer_10/intermediate/dense/Pow/y* 
_output_shapes
:
*
T0
s
.bert/encoder/layer_10/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
¼
,bert/encoder/layer_10/intermediate/dense/mulMul.bert/encoder/layer_10/intermediate/dense/mul/x,bert/encoder/layer_10/intermediate/dense/Pow* 
_output_shapes
:
*
T0
¾
,bert/encoder/layer_10/intermediate/dense/addAdd0bert/encoder/layer_10/intermediate/dense/BiasAdd,bert/encoder/layer_10/intermediate/dense/mul* 
_output_shapes
:
*
T0
u
0bert/encoder/layer_10/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
À
.bert/encoder/layer_10/intermediate/dense/mul_1Mul0bert/encoder/layer_10/intermediate/dense/mul_1/x,bert/encoder/layer_10/intermediate/dense/add*
T0* 
_output_shapes
:


-bert/encoder/layer_10/intermediate/dense/TanhTanh.bert/encoder/layer_10/intermediate/dense/mul_1*
T0* 
_output_shapes
:

u
0bert/encoder/layer_10/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Á
.bert/encoder/layer_10/intermediate/dense/add_1Add0bert/encoder/layer_10/intermediate/dense/add_1/x-bert/encoder/layer_10/intermediate/dense/Tanh* 
_output_shapes
:
*
T0
u
0bert/encoder/layer_10/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Â
.bert/encoder/layer_10/intermediate/dense/mul_2Mul0bert/encoder/layer_10/intermediate/dense/mul_2/x.bert/encoder/layer_10/intermediate/dense/add_1* 
_output_shapes
:
*
T0
Â
.bert/encoder/layer_10/intermediate/dense/mul_3Mul0bert/encoder/layer_10/intermediate/dense/BiasAdd.bert/encoder/layer_10/intermediate/dense/mul_2*
T0* 
_output_shapes
:

Û
Lbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
dtype0*
_output_shapes
:
Î
Kbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/meanConst*
_output_shapes
: *
valueB
 *    *<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
dtype0
Ð
Mbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
dtype0
Æ
Vbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalLbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
seed2 
á
Jbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/mulMulVbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalMbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel* 
_output_shapes
:

Ï
Fbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normalAddJbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/mulKbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal/mean*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel* 
_output_shapes
:

ß
)bert/encoder/layer_10/output/dense/kernel
VariableV2*
shared_name *<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

¿
0bert/encoder/layer_10/output/dense/kernel/AssignAssign)bert/encoder/layer_10/output/dense/kernelFbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Î
.bert/encoder/layer_10/output/dense/kernel/readIdentity)bert/encoder/layer_10/output/dense/kernel* 
_output_shapes
:
*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel
Ä
9bert/encoder/layer_10/output/dense/bias/Initializer/zerosConst*
valueB*    *:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
dtype0*
_output_shapes	
:
Ñ
'bert/encoder/layer_10/output/dense/bias
VariableV2*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
§
.bert/encoder/layer_10/output/dense/bias/AssignAssign'bert/encoder/layer_10/output/dense/bias9bert/encoder/layer_10/output/dense/bias/Initializer/zeros*
T0*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ã
,bert/encoder/layer_10/output/dense/bias/readIdentity'bert/encoder/layer_10/output/dense/bias*
T0*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
_output_shapes	
:
ä
)bert/encoder/layer_10/output/dense/MatMulMatMul.bert/encoder/layer_10/intermediate/dense/mul_3.bert/encoder/layer_10/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
Ð
*bert/encoder/layer_10/output/dense/BiasAddBiasAdd)bert/encoder/layer_10/output/dense/MatMul,bert/encoder/layer_10/output/dense/bias/read* 
_output_shapes
:
*
T0*
data_formatNHWC
À
 bert/encoder/layer_10/output/addAdd*bert/encoder/layer_10/output/dense/BiasAdd@bert/encoder/layer_10/attention/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

Ì
=bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
Ù
+bert/encoder/layer_10/output/LayerNorm/beta
VariableV2*
_output_shapes	
:*
shared_name *>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
	container *
shape:*
dtype0
·
2bert/encoder/layer_10/output/LayerNorm/beta/AssignAssign+bert/encoder/layer_10/output/LayerNorm/beta=bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_10/output/LayerNorm/beta/readIdentity+bert/encoder/layer_10/output/LayerNorm/beta*
T0*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
_output_shapes	
:
Í
=bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
Û
,bert/encoder/layer_10/output/LayerNorm/gamma
VariableV2*
shared_name *?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
º
3bert/encoder/layer_10/output/LayerNorm/gamma/AssignAssign,bert/encoder/layer_10/output/LayerNorm/gamma=bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ò
1bert/encoder/layer_10/output/LayerNorm/gamma/readIdentity,bert/encoder/layer_10/output/LayerNorm/gamma*
T0*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
_output_shapes	
:

Ebert/encoder/layer_10/output/LayerNorm/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
ë
3bert/encoder/layer_10/output/LayerNorm/moments/meanMean bert/encoder/layer_10/output/addEbert/encoder/layer_10/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
ª
;bert/encoder/layer_10/output/LayerNorm/moments/StopGradientStopGradient3bert/encoder/layer_10/output/LayerNorm/moments/mean*
T0*
_output_shapes
:	
ß
@bert/encoder/layer_10/output/LayerNorm/moments/SquaredDifferenceSquaredDifference bert/encoder/layer_10/output/add;bert/encoder/layer_10/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Ibert/encoder/layer_10/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

7bert/encoder/layer_10/output/LayerNorm/moments/varianceMean@bert/encoder/layer_10/output/LayerNorm/moments/SquaredDifferenceIbert/encoder/layer_10/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
{
6bert/encoder/layer_10/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ö
4bert/encoder/layer_10/output/LayerNorm/batchnorm/addAdd7bert/encoder/layer_10/output/LayerNorm/moments/variance6bert/encoder/layer_10/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	

6bert/encoder/layer_10/output/LayerNorm/batchnorm/RsqrtRsqrt4bert/encoder/layer_10/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
Ñ
4bert/encoder/layer_10/output/LayerNorm/batchnorm/mulMul6bert/encoder/layer_10/output/LayerNorm/batchnorm/Rsqrt1bert/encoder/layer_10/output/LayerNorm/gamma/read* 
_output_shapes
:
*
T0
À
6bert/encoder/layer_10/output/LayerNorm/batchnorm/mul_1Mul bert/encoder/layer_10/output/add4bert/encoder/layer_10/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
Ó
6bert/encoder/layer_10/output/LayerNorm/batchnorm/mul_2Mul3bert/encoder/layer_10/output/LayerNorm/moments/mean4bert/encoder/layer_10/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
Ð
4bert/encoder/layer_10/output/LayerNorm/batchnorm/subSub0bert/encoder/layer_10/output/LayerNorm/beta/read6bert/encoder/layer_10/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ö
6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_1Add6bert/encoder/layer_10/output/LayerNorm/batchnorm/mul_14bert/encoder/layer_10/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

ë
Tbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
dtype0*
_output_shapes
:
Þ
Sbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
dtype0*
_output_shapes
: 
à
Ubert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
dtype0*
_output_shapes
: 
Þ
^bert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel

Rbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/mulMul^bert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalUbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/stddev*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel* 
_output_shapes
:
*
T0
ï
Nbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normalAddRbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/mulSbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel* 
_output_shapes
:

ï
1bert/encoder/layer_11/attention/self/query/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
	container *
shape:

ß
8bert/encoder/layer_11/attention/self/query/kernel/AssignAssign1bert/encoder/layer_11/attention/self/query/kernelNbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
æ
6bert/encoder/layer_11/attention/self/query/kernel/readIdentity1bert/encoder/layer_11/attention/self/query/kernel*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel* 
_output_shapes
:

Ô
Abert/encoder/layer_11/attention/self/query/bias/Initializer/zerosConst*
valueB*    *B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
dtype0*
_output_shapes	
:
á
/bert/encoder/layer_11/attention/self/query/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
	container *
shape:
Ç
6bert/encoder/layer_11/attention/self/query/bias/AssignAssign/bert/encoder/layer_11/attention/self/query/biasAbert/encoder/layer_11/attention/self/query/bias/Initializer/zeros*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Û
4bert/encoder/layer_11/attention/self/query/bias/readIdentity/bert/encoder/layer_11/attention/self/query/bias*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
_output_shapes	
:
ü
1bert/encoder/layer_11/attention/self/query/MatMulMatMul6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_16bert/encoder/layer_11/attention/self/query/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
è
2bert/encoder/layer_11/attention/self/query/BiasAddBiasAdd1bert/encoder/layer_11/attention/self/query/MatMul4bert/encoder/layer_11/attention/self/query/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

ç
Rbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
dtype0*
_output_shapes
:
Ú
Qbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel
Ü
Sbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*
_output_shapes
: *
valueB
 *
×£<*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
dtype0
Ø
\bert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/shape*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ù
Pbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/mulMul\bert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/stddev* 
_output_shapes
:
*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel
ç
Lbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normalAddPbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/mulQbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel* 
_output_shapes
:

ë
/bert/encoder/layer_11/attention/self/key/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
	container *
shape:

×
6bert/encoder/layer_11/attention/self/key/kernel/AssignAssign/bert/encoder/layer_11/attention/self/key/kernelLbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel
à
4bert/encoder/layer_11/attention/self/key/kernel/readIdentity/bert/encoder/layer_11/attention/self/key/kernel*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel* 
_output_shapes
:

Ð
?bert/encoder/layer_11/attention/self/key/bias/Initializer/zerosConst*
valueB*    *@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias*
dtype0*
_output_shapes	
:
Ý
-bert/encoder/layer_11/attention/self/key/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias
¿
4bert/encoder/layer_11/attention/self/key/bias/AssignAssign-bert/encoder/layer_11/attention/self/key/bias?bert/encoder/layer_11/attention/self/key/bias/Initializer/zeros*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
Õ
2bert/encoder/layer_11/attention/self/key/bias/readIdentity-bert/encoder/layer_11/attention/self/key/bias*
T0*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias*
_output_shapes	
:
ø
/bert/encoder/layer_11/attention/self/key/MatMulMatMul6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_14bert/encoder/layer_11/attention/self/key/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
â
0bert/encoder/layer_11/attention/self/key/BiasAddBiasAdd/bert/encoder/layer_11/attention/self/key/MatMul2bert/encoder/layer_11/attention/self/key/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

ë
Tbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
dtype0*
_output_shapes
:
Þ
Sbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
dtype0*
_output_shapes
: 
à
Ubert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
dtype0*
_output_shapes
: 
Þ
^bert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalTbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel

Rbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/mulMul^bert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalUbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel* 
_output_shapes
:

ï
Nbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normalAddRbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/mulSbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel* 
_output_shapes
:

ï
1bert/encoder/layer_11/attention/self/value/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel
ß
8bert/encoder/layer_11/attention/self/value/kernel/AssignAssign1bert/encoder/layer_11/attention/self/value/kernelNbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

æ
6bert/encoder/layer_11/attention/self/value/kernel/readIdentity1bert/encoder/layer_11/attention/self/value/kernel* 
_output_shapes
:
*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel
Ô
Abert/encoder/layer_11/attention/self/value/bias/Initializer/zerosConst*
valueB*    *B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
dtype0*
_output_shapes	
:
á
/bert/encoder/layer_11/attention/self/value/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias
Ç
6bert/encoder/layer_11/attention/self/value/bias/AssignAssign/bert/encoder/layer_11/attention/self/value/biasAbert/encoder/layer_11/attention/self/value/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
validate_shape(
Û
4bert/encoder/layer_11/attention/self/value/bias/readIdentity/bert/encoder/layer_11/attention/self/value/bias*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
_output_shapes	
:
ü
1bert/encoder/layer_11/attention/self/value/MatMulMatMul6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_16bert/encoder/layer_11/attention/self/value/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
è
2bert/encoder/layer_11/attention/self/value/BiasAddBiasAdd1bert/encoder/layer_11/attention/self/value/MatMul4bert/encoder/layer_11/attention/self/value/bias/read*
data_formatNHWC* 
_output_shapes
:
*
T0

2bert/encoder/layer_11/attention/self/Reshape/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ß
,bert/encoder/layer_11/attention/self/ReshapeReshape2bert/encoder/layer_11/attention/self/query/BiasAdd2bert/encoder/layer_11/attention/self/Reshape/shape*
T0*
Tshape0*'
_output_shapes
:@

3bert/encoder/layer_11/attention/self/transpose/permConst*
dtype0*
_output_shapes
:*%
valueB"             
Ý
.bert/encoder/layer_11/attention/self/transpose	Transpose,bert/encoder/layer_11/attention/self/Reshape3bert/encoder/layer_11/attention/self/transpose/perm*'
_output_shapes
:@*
Tperm0*
T0

4bert/encoder/layer_11/attention/self/Reshape_1/shapeConst*
_output_shapes
:*%
valueB"         @   *
dtype0
á
.bert/encoder/layer_11/attention/self/Reshape_1Reshape0bert/encoder/layer_11/attention/self/key/BiasAdd4bert/encoder/layer_11/attention/self/Reshape_1/shape*
T0*
Tshape0*'
_output_shapes
:@

5bert/encoder/layer_11/attention/self/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
ã
0bert/encoder/layer_11/attention/self/transpose_1	Transpose.bert/encoder/layer_11/attention/self/Reshape_15bert/encoder/layer_11/attention/self/transpose_1/perm*
T0*'
_output_shapes
:@*
Tperm0
ë
+bert/encoder/layer_11/attention/self/MatMulBatchMatMulV2.bert/encoder/layer_11/attention/self/transpose0bert/encoder/layer_11/attention/self/transpose_1*
adj_x( *
adj_y(*
T0*(
_output_shapes
:
o
*bert/encoder/layer_11/attention/self/Mul/yConst*
valueB
 *   >*
dtype0*
_output_shapes
: 
»
(bert/encoder/layer_11/attention/self/MulMul+bert/encoder/layer_11/attention/self/MatMul*bert/encoder/layer_11/attention/self/Mul/y*(
_output_shapes
:*
T0
}
3bert/encoder/layer_11/attention/self/ExpandDims/dimConst*
valueB:*
dtype0*
_output_shapes
:
Ã
/bert/encoder/layer_11/attention/self/ExpandDims
ExpandDimsbert/encoder/mul3bert/encoder/layer_11/attention/self/ExpandDims/dim*(
_output_shapes
:*

Tdim0*
T0
o
*bert/encoder/layer_11/attention/self/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¿
(bert/encoder/layer_11/attention/self/subSub*bert/encoder/layer_11/attention/self/sub/x/bert/encoder/layer_11/attention/self/ExpandDims*
T0*(
_output_shapes
:
q
,bert/encoder/layer_11/attention/self/mul_1/yConst*
valueB
 * @Æ*
dtype0*
_output_shapes
: 
¼
*bert/encoder/layer_11/attention/self/mul_1Mul(bert/encoder/layer_11/attention/self/sub,bert/encoder/layer_11/attention/self/mul_1/y*(
_output_shapes
:*
T0
¸
(bert/encoder/layer_11/attention/self/addAdd(bert/encoder/layer_11/attention/self/Mul*bert/encoder/layer_11/attention/self/mul_1*
T0*(
_output_shapes
:

,bert/encoder/layer_11/attention/self/SoftmaxSoftmax(bert/encoder/layer_11/attention/self/add*
T0*(
_output_shapes
:

4bert/encoder/layer_11/attention/self/Reshape_2/shapeConst*%
valueB"         @   *
dtype0*
_output_shapes
:
ã
.bert/encoder/layer_11/attention/self/Reshape_2Reshape2bert/encoder/layer_11/attention/self/value/BiasAdd4bert/encoder/layer_11/attention/self/Reshape_2/shape*
T0*
Tshape0*'
_output_shapes
:@

5bert/encoder/layer_11/attention/self/transpose_2/permConst*%
valueB"             *
dtype0*
_output_shapes
:
ã
0bert/encoder/layer_11/attention/self/transpose_2	Transpose.bert/encoder/layer_11/attention/self/Reshape_25bert/encoder/layer_11/attention/self/transpose_2/perm*
Tperm0*
T0*'
_output_shapes
:@
ê
-bert/encoder/layer_11/attention/self/MatMul_1BatchMatMulV2,bert/encoder/layer_11/attention/self/Softmax0bert/encoder/layer_11/attention/self/transpose_2*'
_output_shapes
:@*
adj_x( *
adj_y( *
T0

5bert/encoder/layer_11/attention/self/transpose_3/permConst*%
valueB"             *
dtype0*
_output_shapes
:
â
0bert/encoder/layer_11/attention/self/transpose_3	Transpose-bert/encoder/layer_11/attention/self/MatMul_15bert/encoder/layer_11/attention/self/transpose_3/perm*
T0*'
_output_shapes
:@*
Tperm0

4bert/encoder/layer_11/attention/self/Reshape_3/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
Ú
.bert/encoder/layer_11/attention/self/Reshape_3Reshape0bert/encoder/layer_11/attention/self/transpose_34bert/encoder/layer_11/attention/self/Reshape_3/shape*
T0*
Tshape0* 
_output_shapes
:

ï
Vbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel
â
Ubert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
ä
Wbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
dtype0*
_output_shapes
: 
ä
`bert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalVbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 

Tbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/mulMul`bert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalWbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel* 
_output_shapes
:

÷
Pbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normalAddTbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/mulUbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel* 
_output_shapes
:

ó
3bert/encoder/layer_11/attention/output/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
	container *
shape:

ç
:bert/encoder/layer_11/attention/output/dense/kernel/AssignAssign3bert/encoder/layer_11/attention/output/dense/kernelPbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ì
8bert/encoder/layer_11/attention/output/dense/kernel/readIdentity3bert/encoder/layer_11/attention/output/dense/kernel* 
_output_shapes
:
*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel
Ø
Cbert/encoder/layer_11/attention/output/dense/bias/Initializer/zerosConst*
valueB*    *D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias*
dtype0*
_output_shapes	
:
å
1bert/encoder/layer_11/attention/output/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias*
	container 
Ï
8bert/encoder/layer_11/attention/output/dense/bias/AssignAssign1bert/encoder/layer_11/attention/output/dense/biasCbert/encoder/layer_11/attention/output/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias
á
6bert/encoder/layer_11/attention/output/dense/bias/readIdentity1bert/encoder/layer_11/attention/output/dense/bias*
_output_shapes	
:*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias
ø
3bert/encoder/layer_11/attention/output/dense/MatMulMatMul.bert/encoder/layer_11/attention/self/Reshape_38bert/encoder/layer_11/attention/output/dense/kernel/read* 
_output_shapes
:
*
transpose_a( *
transpose_b( *
T0
î
4bert/encoder/layer_11/attention/output/dense/BiasAddBiasAdd3bert/encoder/layer_11/attention/output/dense/MatMul6bert/encoder/layer_11/attention/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

Ê
*bert/encoder/layer_11/attention/output/addAdd4bert/encoder/layer_11/attention/output/dense/BiasAdd6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_1*
T0* 
_output_shapes
:

à
Gbert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
í
5bert/encoder/layer_11/attention/output/LayerNorm/beta
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
	container 
ß
<bert/encoder/layer_11/attention/output/LayerNorm/beta/AssignAssign5bert/encoder/layer_11/attention/output/LayerNorm/betaGbert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
í
:bert/encoder/layer_11/attention/output/LayerNorm/beta/readIdentity5bert/encoder/layer_11/attention/output/LayerNorm/beta*
T0*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta*
_output_shapes	
:
á
Gbert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/onesConst*
valueB*  ?*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
dtype0*
_output_shapes	
:
ï
6bert/encoder/layer_11/attention/output/LayerNorm/gamma
VariableV2*
shared_name *I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
	container *
shape:*
dtype0*
_output_shapes	
:
â
=bert/encoder/layer_11/attention/output/LayerNorm/gamma/AssignAssign6bert/encoder/layer_11/attention/output/LayerNorm/gammaGbert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/ones*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ð
;bert/encoder/layer_11/attention/output/LayerNorm/gamma/readIdentity6bert/encoder/layer_11/attention/output/LayerNorm/gamma*
_output_shapes	
:*
T0*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma

Obert/encoder/layer_11/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:

=bert/encoder/layer_11/attention/output/LayerNorm/moments/meanMean*bert/encoder/layer_11/attention/output/addObert/encoder/layer_11/attention/output/LayerNorm/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
¾
Ebert/encoder/layer_11/attention/output/LayerNorm/moments/StopGradientStopGradient=bert/encoder/layer_11/attention/output/LayerNorm/moments/mean*
_output_shapes
:	*
T0
ý
Jbert/encoder/layer_11/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference*bert/encoder/layer_11/attention/output/addEbert/encoder/layer_11/attention/output/LayerNorm/moments/StopGradient*
T0* 
_output_shapes
:


Sbert/encoder/layer_11/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
±
Abert/encoder/layer_11/attention/output/LayerNorm/moments/varianceMeanJbert/encoder/layer_11/attention/output/LayerNorm/moments/SquaredDifferenceSbert/encoder/layer_11/attention/output/LayerNorm/moments/variance/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0

@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
ô
>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/addAddAbert/encoder/layer_11/attention/output/LayerNorm/moments/variance@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	
³
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/RsqrtRsqrt>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
ï
>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mulMul@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/Rsqrt;bert/encoder/layer_11/attention/output/LayerNorm/gamma/read*
T0* 
_output_shapes
:

Þ
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul_1Mul*bert/encoder/layer_11/attention/output/add>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
ñ
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul_2Mul=bert/encoder/layer_11/attention/output/LayerNorm/moments/mean>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
î
>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/subSub:bert/encoder/layer_11/attention/output/LayerNorm/beta/read@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

ô
@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add_1Add@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/mul_1>bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/sub*
T0* 
_output_shapes
:

ç
Rbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
dtype0*
_output_shapes
:
Ú
Qbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
dtype0*
_output_shapes
: 
Ü
Sbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *
×£<*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel
Ø
\bert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalRbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel
ù
Pbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/mulMul\bert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalSbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel* 
_output_shapes
:

ç
Lbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normalAddPbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/mulQbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel
ë
/bert/encoder/layer_11/intermediate/dense/kernel
VariableV2*
shared_name *B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

×
6bert/encoder/layer_11/intermediate/dense/kernel/AssignAssign/bert/encoder/layer_11/intermediate/dense/kernelLbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
validate_shape(
à
4bert/encoder/layer_11/intermediate/dense/kernel/readIdentity/bert/encoder/layer_11/intermediate/dense/kernel*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel* 
_output_shapes
:

Ü
Obert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros/shape_as_tensorConst*
valueB:*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
dtype0*
_output_shapes
:
Ì
Ebert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros/ConstConst*
valueB
 *    *@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
dtype0*
_output_shapes
: 
Ù
?bert/encoder/layer_11/intermediate/dense/bias/Initializer/zerosFillObert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros/shape_as_tensorEbert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros/Const*
T0*

index_type0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
_output_shapes	
:
Ý
-bert/encoder/layer_11/intermediate/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
	container *
shape:
¿
4bert/encoder/layer_11/intermediate/dense/bias/AssignAssign-bert/encoder/layer_11/intermediate/dense/bias?bert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
validate_shape(
Õ
2bert/encoder/layer_11/intermediate/dense/bias/readIdentity-bert/encoder/layer_11/intermediate/dense/bias*
T0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
_output_shapes	
:

/bert/encoder/layer_11/intermediate/dense/MatMulMatMul@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add_14bert/encoder/layer_11/intermediate/dense/kernel/read*
T0* 
_output_shapes
:
*
transpose_a( *
transpose_b( 
â
0bert/encoder/layer_11/intermediate/dense/BiasAddBiasAdd/bert/encoder/layer_11/intermediate/dense/MatMul2bert/encoder/layer_11/intermediate/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

s
.bert/encoder/layer_11/intermediate/dense/Pow/yConst*
valueB
 *  @@*
dtype0*
_output_shapes
: 
À
,bert/encoder/layer_11/intermediate/dense/PowPow0bert/encoder/layer_11/intermediate/dense/BiasAdd.bert/encoder/layer_11/intermediate/dense/Pow/y*
T0* 
_output_shapes
:

s
.bert/encoder/layer_11/intermediate/dense/mul/xConst*
valueB
 *'7=*
dtype0*
_output_shapes
: 
¼
,bert/encoder/layer_11/intermediate/dense/mulMul.bert/encoder/layer_11/intermediate/dense/mul/x,bert/encoder/layer_11/intermediate/dense/Pow* 
_output_shapes
:
*
T0
¾
,bert/encoder/layer_11/intermediate/dense/addAdd0bert/encoder/layer_11/intermediate/dense/BiasAdd,bert/encoder/layer_11/intermediate/dense/mul* 
_output_shapes
:
*
T0
u
0bert/encoder/layer_11/intermediate/dense/mul_1/xConst*
valueB
 **BL?*
dtype0*
_output_shapes
: 
À
.bert/encoder/layer_11/intermediate/dense/mul_1Mul0bert/encoder/layer_11/intermediate/dense/mul_1/x,bert/encoder/layer_11/intermediate/dense/add*
T0* 
_output_shapes
:


-bert/encoder/layer_11/intermediate/dense/TanhTanh.bert/encoder/layer_11/intermediate/dense/mul_1*
T0* 
_output_shapes
:

u
0bert/encoder/layer_11/intermediate/dense/add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Á
.bert/encoder/layer_11/intermediate/dense/add_1Add0bert/encoder/layer_11/intermediate/dense/add_1/x-bert/encoder/layer_11/intermediate/dense/Tanh* 
_output_shapes
:
*
T0
u
0bert/encoder/layer_11/intermediate/dense/mul_2/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
Â
.bert/encoder/layer_11/intermediate/dense/mul_2Mul0bert/encoder/layer_11/intermediate/dense/mul_2/x.bert/encoder/layer_11/intermediate/dense/add_1* 
_output_shapes
:
*
T0
Â
.bert/encoder/layer_11/intermediate/dense/mul_3Mul0bert/encoder/layer_11/intermediate/dense/BiasAdd.bert/encoder/layer_11/intermediate/dense/mul_2*
T0* 
_output_shapes
:

Û
Lbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
dtype0*
_output_shapes
:
Î
Kbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
dtype0*
_output_shapes
: 
Ð
Mbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
dtype0*
_output_shapes
: 
Æ
Vbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalLbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/shape*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
á
Jbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/mulMulVbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalMbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel* 
_output_shapes
:

Ï
Fbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normalAddJbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/mulKbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal/mean* 
_output_shapes
:
*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel
ß
)bert/encoder/layer_11/output/dense/kernel
VariableV2*
shared_name *<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

¿
0bert/encoder/layer_11/output/dense/kernel/AssignAssign)bert/encoder/layer_11/output/dense/kernelFbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Î
.bert/encoder/layer_11/output/dense/kernel/readIdentity)bert/encoder/layer_11/output/dense/kernel* 
_output_shapes
:
*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel
Ä
9bert/encoder/layer_11/output/dense/bias/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
dtype0
Ñ
'bert/encoder/layer_11/output/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
	container 
§
.bert/encoder/layer_11/output/dense/bias/AssignAssign'bert/encoder/layer_11/output/dense/bias9bert/encoder/layer_11/output/dense/bias/Initializer/zeros*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
validate_shape(*
_output_shapes	
:
Ã
,bert/encoder/layer_11/output/dense/bias/readIdentity'bert/encoder/layer_11/output/dense/bias*
T0*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
_output_shapes	
:
ä
)bert/encoder/layer_11/output/dense/MatMulMatMul.bert/encoder/layer_11/intermediate/dense/mul_3.bert/encoder/layer_11/output/dense/kernel/read*
transpose_b( *
T0* 
_output_shapes
:
*
transpose_a( 
Ð
*bert/encoder/layer_11/output/dense/BiasAddBiasAdd)bert/encoder/layer_11/output/dense/MatMul,bert/encoder/layer_11/output/dense/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:

À
 bert/encoder/layer_11/output/addAdd*bert/encoder/layer_11/output/dense/BiasAdd@bert/encoder/layer_11/attention/output/LayerNorm/batchnorm/add_1* 
_output_shapes
:
*
T0
Ì
=bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zerosConst*
valueB*    *>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
dtype0*
_output_shapes	
:
Ù
+bert/encoder/layer_11/output/LayerNorm/beta
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
	container *
shape:
·
2bert/encoder/layer_11/output/LayerNorm/beta/AssignAssign+bert/encoder/layer_11/output/LayerNorm/beta=bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zeros*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
Ï
0bert/encoder/layer_11/output/LayerNorm/beta/readIdentity+bert/encoder/layer_11/output/LayerNorm/beta*
_output_shapes	
:*
T0*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta
Í
=bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/onesConst*
dtype0*
_output_shapes	
:*
valueB*  ?*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma
Û
,bert/encoder/layer_11/output/LayerNorm/gamma
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma
º
3bert/encoder/layer_11/output/LayerNorm/gamma/AssignAssign,bert/encoder/layer_11/output/LayerNorm/gamma=bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/ones*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ò
1bert/encoder/layer_11/output/LayerNorm/gamma/readIdentity,bert/encoder/layer_11/output/LayerNorm/gamma*
T0*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
_output_shapes	
:

Ebert/encoder/layer_11/output/LayerNorm/moments/mean/reduction_indicesConst*
valueB:*
dtype0*
_output_shapes
:
ë
3bert/encoder/layer_11/output/LayerNorm/moments/meanMean bert/encoder/layer_11/output/addEbert/encoder/layer_11/output/LayerNorm/moments/mean/reduction_indices*
_output_shapes
:	*
	keep_dims(*

Tidx0*
T0
ª
;bert/encoder/layer_11/output/LayerNorm/moments/StopGradientStopGradient3bert/encoder/layer_11/output/LayerNorm/moments/mean*
_output_shapes
:	*
T0
ß
@bert/encoder/layer_11/output/LayerNorm/moments/SquaredDifferenceSquaredDifference bert/encoder/layer_11/output/add;bert/encoder/layer_11/output/LayerNorm/moments/StopGradient* 
_output_shapes
:
*
T0

Ibert/encoder/layer_11/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
valueB:*
dtype0

7bert/encoder/layer_11/output/LayerNorm/moments/varianceMean@bert/encoder/layer_11/output/LayerNorm/moments/SquaredDifferenceIbert/encoder/layer_11/output/LayerNorm/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*
_output_shapes
:	
{
6bert/encoder/layer_11/output/LayerNorm/batchnorm/add/yConst*
valueB
 *Ì¼+*
dtype0*
_output_shapes
: 
Ö
4bert/encoder/layer_11/output/LayerNorm/batchnorm/addAdd7bert/encoder/layer_11/output/LayerNorm/moments/variance6bert/encoder/layer_11/output/LayerNorm/batchnorm/add/y*
T0*
_output_shapes
:	

6bert/encoder/layer_11/output/LayerNorm/batchnorm/RsqrtRsqrt4bert/encoder/layer_11/output/LayerNorm/batchnorm/add*
T0*
_output_shapes
:	
Ñ
4bert/encoder/layer_11/output/LayerNorm/batchnorm/mulMul6bert/encoder/layer_11/output/LayerNorm/batchnorm/Rsqrt1bert/encoder/layer_11/output/LayerNorm/gamma/read* 
_output_shapes
:
*
T0
À
6bert/encoder/layer_11/output/LayerNorm/batchnorm/mul_1Mul bert/encoder/layer_11/output/add4bert/encoder/layer_11/output/LayerNorm/batchnorm/mul* 
_output_shapes
:
*
T0
Ó
6bert/encoder/layer_11/output/LayerNorm/batchnorm/mul_2Mul3bert/encoder/layer_11/output/LayerNorm/moments/mean4bert/encoder/layer_11/output/LayerNorm/batchnorm/mul*
T0* 
_output_shapes
:

Ð
4bert/encoder/layer_11/output/LayerNorm/batchnorm/subSub0bert/encoder/layer_11/output/LayerNorm/beta/read6bert/encoder/layer_11/output/LayerNorm/batchnorm/mul_2*
T0* 
_output_shapes
:

Ö
6bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1Add6bert/encoder/layer_11/output/LayerNorm/batchnorm/mul_14bert/encoder/layer_11/output/LayerNorm/batchnorm/sub* 
_output_shapes
:
*
T0
q
bert/encoder/Reshape_2/shapeConst*
dtype0*
_output_shapes
:*!
valueB"         
³
bert/encoder/Reshape_2Reshape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_2/shape*
Tshape0*$
_output_shapes
:*
T0
q
bert/encoder/Reshape_3/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
³
bert/encoder/Reshape_3Reshape5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_3/shape*$
_output_shapes
:*
T0*
Tshape0
q
bert/encoder/Reshape_4/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
³
bert/encoder/Reshape_4Reshape5bert/encoder/layer_2/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_4/shape*
T0*
Tshape0*$
_output_shapes
:
q
bert/encoder/Reshape_5/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
³
bert/encoder/Reshape_5Reshape5bert/encoder/layer_3/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_5/shape*$
_output_shapes
:*
T0*
Tshape0
q
bert/encoder/Reshape_6/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
³
bert/encoder/Reshape_6Reshape5bert/encoder/layer_4/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_6/shape*
T0*
Tshape0*$
_output_shapes
:
q
bert/encoder/Reshape_7/shapeConst*
_output_shapes
:*!
valueB"         *
dtype0
³
bert/encoder/Reshape_7Reshape5bert/encoder/layer_5/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_7/shape*$
_output_shapes
:*
T0*
Tshape0
q
bert/encoder/Reshape_8/shapeConst*
dtype0*
_output_shapes
:*!
valueB"         
³
bert/encoder/Reshape_8Reshape5bert/encoder/layer_6/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_8/shape*
T0*
Tshape0*$
_output_shapes
:
q
bert/encoder/Reshape_9/shapeConst*
dtype0*
_output_shapes
:*!
valueB"         
³
bert/encoder/Reshape_9Reshape5bert/encoder/layer_7/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_9/shape*$
_output_shapes
:*
T0*
Tshape0
r
bert/encoder/Reshape_10/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
µ
bert/encoder/Reshape_10Reshape5bert/encoder/layer_8/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_10/shape*$
_output_shapes
:*
T0*
Tshape0
r
bert/encoder/Reshape_11/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
µ
bert/encoder/Reshape_11Reshape5bert/encoder/layer_9/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_11/shape*
Tshape0*$
_output_shapes
:*
T0
r
bert/encoder/Reshape_12/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
¶
bert/encoder/Reshape_12Reshape6bert/encoder/layer_10/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_12/shape*$
_output_shapes
:*
T0*
Tshape0
r
bert/encoder/Reshape_13/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
¶
bert/encoder/Reshape_13Reshape6bert/encoder/layer_11/output/LayerNorm/batchnorm/add_1bert/encoder/Reshape_13/shape*
T0*
Tshape0*$
_output_shapes
:
t
bert/pooler/strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
v
!bert/pooler/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"           
v
!bert/pooler/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
È
bert/pooler/strided_sliceStridedSlicebert/encoder/Reshape_13bert/pooler/strided_slice/stack!bert/pooler/strided_slice/stack_1!bert/pooler/strided_slice/stack_2*
end_mask*#
_output_shapes
:*
T0*
Index0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask 
z
bert/pooler/SqueezeSqueezebert/pooler/strided_slice*
T0*
_output_shapes
:	*
squeeze_dims

¹
;bert/pooler/dense/kernel/Initializer/truncated_normal/shapeConst*
valueB"      *+
_class!
loc:@bert/pooler/dense/kernel*
dtype0*
_output_shapes
:
¬
:bert/pooler/dense/kernel/Initializer/truncated_normal/meanConst*
valueB
 *    *+
_class!
loc:@bert/pooler/dense/kernel*
dtype0*
_output_shapes
: 
®
<bert/pooler/dense/kernel/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*+
_class!
loc:@bert/pooler/dense/kernel*
dtype0*
_output_shapes
: 

Ebert/pooler/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;bert/pooler/dense/kernel/Initializer/truncated_normal/shape*

seed *
T0*+
_class!
loc:@bert/pooler/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:


9bert/pooler/dense/kernel/Initializer/truncated_normal/mulMulEbert/pooler/dense/kernel/Initializer/truncated_normal/TruncatedNormal<bert/pooler/dense/kernel/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:


5bert/pooler/dense/kernel/Initializer/truncated_normalAdd9bert/pooler/dense/kernel/Initializer/truncated_normal/mul:bert/pooler/dense/kernel/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:

½
bert/pooler/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *+
_class!
loc:@bert/pooler/dense/kernel*
	container *
shape:

û
bert/pooler/dense/kernel/AssignAssignbert/pooler/dense/kernel5bert/pooler/dense/kernel/Initializer/truncated_normal* 
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@bert/pooler/dense/kernel*
validate_shape(

bert/pooler/dense/kernel/readIdentitybert/pooler/dense/kernel* 
_output_shapes
:
*
T0*+
_class!
loc:@bert/pooler/dense/kernel
¢
(bert/pooler/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *)
_class
loc:@bert/pooler/dense/bias
¯
bert/pooler/dense/bias
VariableV2*
shared_name *)
_class
loc:@bert/pooler/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ã
bert/pooler/dense/bias/AssignAssignbert/pooler/dense/bias(bert/pooler/dense/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@bert/pooler/dense/bias*
validate_shape(*
_output_shapes	
:

bert/pooler/dense/bias/readIdentitybert/pooler/dense/bias*
T0*)
_class
loc:@bert/pooler/dense/bias*
_output_shapes	
:
¦
bert/pooler/dense/MatMulMatMulbert/pooler/Squeezebert/pooler/dense/kernel/read*
_output_shapes
:	*
transpose_a( *
transpose_b( *
T0

bert/pooler/dense/BiasAddBiasAddbert/pooler/dense/MatMulbert/pooler/dense/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	
c
bert/pooler/dense/TanhTanhbert/pooler/dense/BiasAdd*
T0*
_output_shapes
:	
¥
1output_weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"      *!
_class
loc:@output_weights

0output_weights/Initializer/truncated_normal/meanConst*
valueB
 *    *!
_class
loc:@output_weights*
dtype0*
_output_shapes
: 

2output_weights/Initializer/truncated_normal/stddevConst*
valueB
 *
×£<*!
_class
loc:@output_weights*
dtype0*
_output_shapes
: 
ô
;output_weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal1output_weights/Initializer/truncated_normal/shape*

seed *
T0*!
_class
loc:@output_weights*
seed2 *
dtype0*
_output_shapes
:	
ô
/output_weights/Initializer/truncated_normal/mulMul;output_weights/Initializer/truncated_normal/TruncatedNormal2output_weights/Initializer/truncated_normal/stddev*
T0*!
_class
loc:@output_weights*
_output_shapes
:	
â
+output_weights/Initializer/truncated_normalAdd/output_weights/Initializer/truncated_normal/mul0output_weights/Initializer/truncated_normal/mean*
T0*!
_class
loc:@output_weights*
_output_shapes
:	
§
output_weights
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *!
_class
loc:@output_weights*
	container *
shape:	
Ò
output_weights/AssignAssignoutput_weights+output_weights/Initializer/truncated_normal*
use_locking(*
T0*!
_class
loc:@output_weights*
validate_shape(*
_output_shapes
:	
|
output_weights/readIdentityoutput_weights*
T0*!
_class
loc:@output_weights*
_output_shapes
:	

output_bias/Initializer/zerosConst*
valueB*    *
_class
loc:@output_bias*
dtype0*
_output_shapes
:

output_bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
_class
loc:@output_bias*
	container *
shape:
¶
output_bias/AssignAssignoutput_biasoutput_bias/Initializer/zeros*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@output_bias*
validate_shape(
n
output_bias/readIdentityoutput_bias*
_output_shapes
:*
T0*
_class
loc:@output_bias

loss/MatMulMatMulbert/pooler/dense/Tanhoutput_weights/read*
T0*
_output_shapes

:*
transpose_a( *
transpose_b(
v
loss/BiasAddBiasAddloss/MatMuloutput_bias/read*
data_formatNHWC*
_output_shapes

:*
T0
N
loss/SoftmaxSoftmaxloss/BiasAdd*
_output_shapes

:*
T0
T
loss/LogSoftmax
LogSoftmaxloss/BiasAdd*
T0*
_output_shapes

:
Z
loss/one_hot/on_valueConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
[
loss/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
T
loss/one_hot/depthConst*
_output_shapes
: *
value	B :*
dtype0
¬
loss/one_hotOneHot	label_idsloss/one_hot/depthloss/one_hot/on_valueloss/one_hot/off_value*
T0*
axisÿÿÿÿÿÿÿÿÿ*
TI0*
_output_shapes

:
W
loss/mulMulloss/one_hotloss/LogSoftmax*
_output_shapes

:*
T0
e
loss/Sum/reduction_indicesConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 
w
loss/SumSumloss/mulloss/Sum/reduction_indices*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
>
loss/NegNegloss/Sum*
T0*
_output_shapes
:
T

loss/ConstConst*
valueB: *
dtype0*
_output_shapes
:
e
	loss/MeanMeanloss/Neg
loss/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
shape: *
dtype0
éJ
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes	
:É*J
valueJBJÉBbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelB5bert/encoder/layer_10/attention/output/LayerNorm/betaB6bert/encoder/layer_10/attention/output/LayerNorm/gammaB1bert/encoder/layer_10/attention/output/dense/biasB3bert/encoder/layer_10/attention/output/dense/kernelB-bert/encoder/layer_10/attention/self/key/biasB/bert/encoder/layer_10/attention/self/key/kernelB/bert/encoder/layer_10/attention/self/query/biasB1bert/encoder/layer_10/attention/self/query/kernelB/bert/encoder/layer_10/attention/self/value/biasB1bert/encoder/layer_10/attention/self/value/kernelB-bert/encoder/layer_10/intermediate/dense/biasB/bert/encoder/layer_10/intermediate/dense/kernelB+bert/encoder/layer_10/output/LayerNorm/betaB,bert/encoder/layer_10/output/LayerNorm/gammaB'bert/encoder/layer_10/output/dense/biasB)bert/encoder/layer_10/output/dense/kernelB5bert/encoder/layer_11/attention/output/LayerNorm/betaB6bert/encoder/layer_11/attention/output/LayerNorm/gammaB1bert/encoder/layer_11/attention/output/dense/biasB3bert/encoder/layer_11/attention/output/dense/kernelB-bert/encoder/layer_11/attention/self/key/biasB/bert/encoder/layer_11/attention/self/key/kernelB/bert/encoder/layer_11/attention/self/query/biasB1bert/encoder/layer_11/attention/self/query/kernelB/bert/encoder/layer_11/attention/self/value/biasB1bert/encoder/layer_11/attention/self/value/kernelB-bert/encoder/layer_11/intermediate/dense/biasB/bert/encoder/layer_11/intermediate/dense/kernelB+bert/encoder/layer_11/output/LayerNorm/betaB,bert/encoder/layer_11/output/LayerNorm/gammaB'bert/encoder/layer_11/output/dense/biasB)bert/encoder/layer_11/output/dense/kernelB4bert/encoder/layer_2/attention/output/LayerNorm/betaB5bert/encoder/layer_2/attention/output/LayerNorm/gammaB0bert/encoder/layer_2/attention/output/dense/biasB2bert/encoder/layer_2/attention/output/dense/kernelB,bert/encoder/layer_2/attention/self/key/biasB.bert/encoder/layer_2/attention/self/key/kernelB.bert/encoder/layer_2/attention/self/query/biasB0bert/encoder/layer_2/attention/self/query/kernelB.bert/encoder/layer_2/attention/self/value/biasB0bert/encoder/layer_2/attention/self/value/kernelB,bert/encoder/layer_2/intermediate/dense/biasB.bert/encoder/layer_2/intermediate/dense/kernelB*bert/encoder/layer_2/output/LayerNorm/betaB+bert/encoder/layer_2/output/LayerNorm/gammaB&bert/encoder/layer_2/output/dense/biasB(bert/encoder/layer_2/output/dense/kernelB4bert/encoder/layer_3/attention/output/LayerNorm/betaB5bert/encoder/layer_3/attention/output/LayerNorm/gammaB0bert/encoder/layer_3/attention/output/dense/biasB2bert/encoder/layer_3/attention/output/dense/kernelB,bert/encoder/layer_3/attention/self/key/biasB.bert/encoder/layer_3/attention/self/key/kernelB.bert/encoder/layer_3/attention/self/query/biasB0bert/encoder/layer_3/attention/self/query/kernelB.bert/encoder/layer_3/attention/self/value/biasB0bert/encoder/layer_3/attention/self/value/kernelB,bert/encoder/layer_3/intermediate/dense/biasB.bert/encoder/layer_3/intermediate/dense/kernelB*bert/encoder/layer_3/output/LayerNorm/betaB+bert/encoder/layer_3/output/LayerNorm/gammaB&bert/encoder/layer_3/output/dense/biasB(bert/encoder/layer_3/output/dense/kernelB4bert/encoder/layer_4/attention/output/LayerNorm/betaB5bert/encoder/layer_4/attention/output/LayerNorm/gammaB0bert/encoder/layer_4/attention/output/dense/biasB2bert/encoder/layer_4/attention/output/dense/kernelB,bert/encoder/layer_4/attention/self/key/biasB.bert/encoder/layer_4/attention/self/key/kernelB.bert/encoder/layer_4/attention/self/query/biasB0bert/encoder/layer_4/attention/self/query/kernelB.bert/encoder/layer_4/attention/self/value/biasB0bert/encoder/layer_4/attention/self/value/kernelB,bert/encoder/layer_4/intermediate/dense/biasB.bert/encoder/layer_4/intermediate/dense/kernelB*bert/encoder/layer_4/output/LayerNorm/betaB+bert/encoder/layer_4/output/LayerNorm/gammaB&bert/encoder/layer_4/output/dense/biasB(bert/encoder/layer_4/output/dense/kernelB4bert/encoder/layer_5/attention/output/LayerNorm/betaB5bert/encoder/layer_5/attention/output/LayerNorm/gammaB0bert/encoder/layer_5/attention/output/dense/biasB2bert/encoder/layer_5/attention/output/dense/kernelB,bert/encoder/layer_5/attention/self/key/biasB.bert/encoder/layer_5/attention/self/key/kernelB.bert/encoder/layer_5/attention/self/query/biasB0bert/encoder/layer_5/attention/self/query/kernelB.bert/encoder/layer_5/attention/self/value/biasB0bert/encoder/layer_5/attention/self/value/kernelB,bert/encoder/layer_5/intermediate/dense/biasB.bert/encoder/layer_5/intermediate/dense/kernelB*bert/encoder/layer_5/output/LayerNorm/betaB+bert/encoder/layer_5/output/LayerNorm/gammaB&bert/encoder/layer_5/output/dense/biasB(bert/encoder/layer_5/output/dense/kernelB4bert/encoder/layer_6/attention/output/LayerNorm/betaB5bert/encoder/layer_6/attention/output/LayerNorm/gammaB0bert/encoder/layer_6/attention/output/dense/biasB2bert/encoder/layer_6/attention/output/dense/kernelB,bert/encoder/layer_6/attention/self/key/biasB.bert/encoder/layer_6/attention/self/key/kernelB.bert/encoder/layer_6/attention/self/query/biasB0bert/encoder/layer_6/attention/self/query/kernelB.bert/encoder/layer_6/attention/self/value/biasB0bert/encoder/layer_6/attention/self/value/kernelB,bert/encoder/layer_6/intermediate/dense/biasB.bert/encoder/layer_6/intermediate/dense/kernelB*bert/encoder/layer_6/output/LayerNorm/betaB+bert/encoder/layer_6/output/LayerNorm/gammaB&bert/encoder/layer_6/output/dense/biasB(bert/encoder/layer_6/output/dense/kernelB4bert/encoder/layer_7/attention/output/LayerNorm/betaB5bert/encoder/layer_7/attention/output/LayerNorm/gammaB0bert/encoder/layer_7/attention/output/dense/biasB2bert/encoder/layer_7/attention/output/dense/kernelB,bert/encoder/layer_7/attention/self/key/biasB.bert/encoder/layer_7/attention/self/key/kernelB.bert/encoder/layer_7/attention/self/query/biasB0bert/encoder/layer_7/attention/self/query/kernelB.bert/encoder/layer_7/attention/self/value/biasB0bert/encoder/layer_7/attention/self/value/kernelB,bert/encoder/layer_7/intermediate/dense/biasB.bert/encoder/layer_7/intermediate/dense/kernelB*bert/encoder/layer_7/output/LayerNorm/betaB+bert/encoder/layer_7/output/LayerNorm/gammaB&bert/encoder/layer_7/output/dense/biasB(bert/encoder/layer_7/output/dense/kernelB4bert/encoder/layer_8/attention/output/LayerNorm/betaB5bert/encoder/layer_8/attention/output/LayerNorm/gammaB0bert/encoder/layer_8/attention/output/dense/biasB2bert/encoder/layer_8/attention/output/dense/kernelB,bert/encoder/layer_8/attention/self/key/biasB.bert/encoder/layer_8/attention/self/key/kernelB.bert/encoder/layer_8/attention/self/query/biasB0bert/encoder/layer_8/attention/self/query/kernelB.bert/encoder/layer_8/attention/self/value/biasB0bert/encoder/layer_8/attention/self/value/kernelB,bert/encoder/layer_8/intermediate/dense/biasB.bert/encoder/layer_8/intermediate/dense/kernelB*bert/encoder/layer_8/output/LayerNorm/betaB+bert/encoder/layer_8/output/LayerNorm/gammaB&bert/encoder/layer_8/output/dense/biasB(bert/encoder/layer_8/output/dense/kernelB4bert/encoder/layer_9/attention/output/LayerNorm/betaB5bert/encoder/layer_9/attention/output/LayerNorm/gammaB0bert/encoder/layer_9/attention/output/dense/biasB2bert/encoder/layer_9/attention/output/dense/kernelB,bert/encoder/layer_9/attention/self/key/biasB.bert/encoder/layer_9/attention/self/key/kernelB.bert/encoder/layer_9/attention/self/query/biasB0bert/encoder/layer_9/attention/self/query/kernelB.bert/encoder/layer_9/attention/self/value/biasB0bert/encoder/layer_9/attention/self/value/kernelB,bert/encoder/layer_9/intermediate/dense/biasB.bert/encoder/layer_9/intermediate/dense/kernelB*bert/encoder/layer_9/output/LayerNorm/betaB+bert/encoder/layer_9/output/LayerNorm/gammaB&bert/encoder/layer_9/output/dense/biasB(bert/encoder/layer_9/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBoutput_biasBoutput_weights
ú
save/SaveV2/shape_and_slicesConst*¨
valueBÉB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:É
»L
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbert/embeddings/LayerNorm/betabert/embeddings/LayerNorm/gamma#bert/embeddings/position_embeddings%bert/embeddings/token_type_embeddingsbert/embeddings/word_embeddings4bert/encoder/layer_0/attention/output/LayerNorm/beta5bert/encoder/layer_0/attention/output/LayerNorm/gamma0bert/encoder/layer_0/attention/output/dense/bias2bert/encoder/layer_0/attention/output/dense/kernel,bert/encoder/layer_0/attention/self/key/bias.bert/encoder/layer_0/attention/self/key/kernel.bert/encoder/layer_0/attention/self/query/bias0bert/encoder/layer_0/attention/self/query/kernel.bert/encoder/layer_0/attention/self/value/bias0bert/encoder/layer_0/attention/self/value/kernel,bert/encoder/layer_0/intermediate/dense/bias.bert/encoder/layer_0/intermediate/dense/kernel*bert/encoder/layer_0/output/LayerNorm/beta+bert/encoder/layer_0/output/LayerNorm/gamma&bert/encoder/layer_0/output/dense/bias(bert/encoder/layer_0/output/dense/kernel4bert/encoder/layer_1/attention/output/LayerNorm/beta5bert/encoder/layer_1/attention/output/LayerNorm/gamma0bert/encoder/layer_1/attention/output/dense/bias2bert/encoder/layer_1/attention/output/dense/kernel,bert/encoder/layer_1/attention/self/key/bias.bert/encoder/layer_1/attention/self/key/kernel.bert/encoder/layer_1/attention/self/query/bias0bert/encoder/layer_1/attention/self/query/kernel.bert/encoder/layer_1/attention/self/value/bias0bert/encoder/layer_1/attention/self/value/kernel,bert/encoder/layer_1/intermediate/dense/bias.bert/encoder/layer_1/intermediate/dense/kernel*bert/encoder/layer_1/output/LayerNorm/beta+bert/encoder/layer_1/output/LayerNorm/gamma&bert/encoder/layer_1/output/dense/bias(bert/encoder/layer_1/output/dense/kernel5bert/encoder/layer_10/attention/output/LayerNorm/beta6bert/encoder/layer_10/attention/output/LayerNorm/gamma1bert/encoder/layer_10/attention/output/dense/bias3bert/encoder/layer_10/attention/output/dense/kernel-bert/encoder/layer_10/attention/self/key/bias/bert/encoder/layer_10/attention/self/key/kernel/bert/encoder/layer_10/attention/self/query/bias1bert/encoder/layer_10/attention/self/query/kernel/bert/encoder/layer_10/attention/self/value/bias1bert/encoder/layer_10/attention/self/value/kernel-bert/encoder/layer_10/intermediate/dense/bias/bert/encoder/layer_10/intermediate/dense/kernel+bert/encoder/layer_10/output/LayerNorm/beta,bert/encoder/layer_10/output/LayerNorm/gamma'bert/encoder/layer_10/output/dense/bias)bert/encoder/layer_10/output/dense/kernel5bert/encoder/layer_11/attention/output/LayerNorm/beta6bert/encoder/layer_11/attention/output/LayerNorm/gamma1bert/encoder/layer_11/attention/output/dense/bias3bert/encoder/layer_11/attention/output/dense/kernel-bert/encoder/layer_11/attention/self/key/bias/bert/encoder/layer_11/attention/self/key/kernel/bert/encoder/layer_11/attention/self/query/bias1bert/encoder/layer_11/attention/self/query/kernel/bert/encoder/layer_11/attention/self/value/bias1bert/encoder/layer_11/attention/self/value/kernel-bert/encoder/layer_11/intermediate/dense/bias/bert/encoder/layer_11/intermediate/dense/kernel+bert/encoder/layer_11/output/LayerNorm/beta,bert/encoder/layer_11/output/LayerNorm/gamma'bert/encoder/layer_11/output/dense/bias)bert/encoder/layer_11/output/dense/kernel4bert/encoder/layer_2/attention/output/LayerNorm/beta5bert/encoder/layer_2/attention/output/LayerNorm/gamma0bert/encoder/layer_2/attention/output/dense/bias2bert/encoder/layer_2/attention/output/dense/kernel,bert/encoder/layer_2/attention/self/key/bias.bert/encoder/layer_2/attention/self/key/kernel.bert/encoder/layer_2/attention/self/query/bias0bert/encoder/layer_2/attention/self/query/kernel.bert/encoder/layer_2/attention/self/value/bias0bert/encoder/layer_2/attention/self/value/kernel,bert/encoder/layer_2/intermediate/dense/bias.bert/encoder/layer_2/intermediate/dense/kernel*bert/encoder/layer_2/output/LayerNorm/beta+bert/encoder/layer_2/output/LayerNorm/gamma&bert/encoder/layer_2/output/dense/bias(bert/encoder/layer_2/output/dense/kernel4bert/encoder/layer_3/attention/output/LayerNorm/beta5bert/encoder/layer_3/attention/output/LayerNorm/gamma0bert/encoder/layer_3/attention/output/dense/bias2bert/encoder/layer_3/attention/output/dense/kernel,bert/encoder/layer_3/attention/self/key/bias.bert/encoder/layer_3/attention/self/key/kernel.bert/encoder/layer_3/attention/self/query/bias0bert/encoder/layer_3/attention/self/query/kernel.bert/encoder/layer_3/attention/self/value/bias0bert/encoder/layer_3/attention/self/value/kernel,bert/encoder/layer_3/intermediate/dense/bias.bert/encoder/layer_3/intermediate/dense/kernel*bert/encoder/layer_3/output/LayerNorm/beta+bert/encoder/layer_3/output/LayerNorm/gamma&bert/encoder/layer_3/output/dense/bias(bert/encoder/layer_3/output/dense/kernel4bert/encoder/layer_4/attention/output/LayerNorm/beta5bert/encoder/layer_4/attention/output/LayerNorm/gamma0bert/encoder/layer_4/attention/output/dense/bias2bert/encoder/layer_4/attention/output/dense/kernel,bert/encoder/layer_4/attention/self/key/bias.bert/encoder/layer_4/attention/self/key/kernel.bert/encoder/layer_4/attention/self/query/bias0bert/encoder/layer_4/attention/self/query/kernel.bert/encoder/layer_4/attention/self/value/bias0bert/encoder/layer_4/attention/self/value/kernel,bert/encoder/layer_4/intermediate/dense/bias.bert/encoder/layer_4/intermediate/dense/kernel*bert/encoder/layer_4/output/LayerNorm/beta+bert/encoder/layer_4/output/LayerNorm/gamma&bert/encoder/layer_4/output/dense/bias(bert/encoder/layer_4/output/dense/kernel4bert/encoder/layer_5/attention/output/LayerNorm/beta5bert/encoder/layer_5/attention/output/LayerNorm/gamma0bert/encoder/layer_5/attention/output/dense/bias2bert/encoder/layer_5/attention/output/dense/kernel,bert/encoder/layer_5/attention/self/key/bias.bert/encoder/layer_5/attention/self/key/kernel.bert/encoder/layer_5/attention/self/query/bias0bert/encoder/layer_5/attention/self/query/kernel.bert/encoder/layer_5/attention/self/value/bias0bert/encoder/layer_5/attention/self/value/kernel,bert/encoder/layer_5/intermediate/dense/bias.bert/encoder/layer_5/intermediate/dense/kernel*bert/encoder/layer_5/output/LayerNorm/beta+bert/encoder/layer_5/output/LayerNorm/gamma&bert/encoder/layer_5/output/dense/bias(bert/encoder/layer_5/output/dense/kernel4bert/encoder/layer_6/attention/output/LayerNorm/beta5bert/encoder/layer_6/attention/output/LayerNorm/gamma0bert/encoder/layer_6/attention/output/dense/bias2bert/encoder/layer_6/attention/output/dense/kernel,bert/encoder/layer_6/attention/self/key/bias.bert/encoder/layer_6/attention/self/key/kernel.bert/encoder/layer_6/attention/self/query/bias0bert/encoder/layer_6/attention/self/query/kernel.bert/encoder/layer_6/attention/self/value/bias0bert/encoder/layer_6/attention/self/value/kernel,bert/encoder/layer_6/intermediate/dense/bias.bert/encoder/layer_6/intermediate/dense/kernel*bert/encoder/layer_6/output/LayerNorm/beta+bert/encoder/layer_6/output/LayerNorm/gamma&bert/encoder/layer_6/output/dense/bias(bert/encoder/layer_6/output/dense/kernel4bert/encoder/layer_7/attention/output/LayerNorm/beta5bert/encoder/layer_7/attention/output/LayerNorm/gamma0bert/encoder/layer_7/attention/output/dense/bias2bert/encoder/layer_7/attention/output/dense/kernel,bert/encoder/layer_7/attention/self/key/bias.bert/encoder/layer_7/attention/self/key/kernel.bert/encoder/layer_7/attention/self/query/bias0bert/encoder/layer_7/attention/self/query/kernel.bert/encoder/layer_7/attention/self/value/bias0bert/encoder/layer_7/attention/self/value/kernel,bert/encoder/layer_7/intermediate/dense/bias.bert/encoder/layer_7/intermediate/dense/kernel*bert/encoder/layer_7/output/LayerNorm/beta+bert/encoder/layer_7/output/LayerNorm/gamma&bert/encoder/layer_7/output/dense/bias(bert/encoder/layer_7/output/dense/kernel4bert/encoder/layer_8/attention/output/LayerNorm/beta5bert/encoder/layer_8/attention/output/LayerNorm/gamma0bert/encoder/layer_8/attention/output/dense/bias2bert/encoder/layer_8/attention/output/dense/kernel,bert/encoder/layer_8/attention/self/key/bias.bert/encoder/layer_8/attention/self/key/kernel.bert/encoder/layer_8/attention/self/query/bias0bert/encoder/layer_8/attention/self/query/kernel.bert/encoder/layer_8/attention/self/value/bias0bert/encoder/layer_8/attention/self/value/kernel,bert/encoder/layer_8/intermediate/dense/bias.bert/encoder/layer_8/intermediate/dense/kernel*bert/encoder/layer_8/output/LayerNorm/beta+bert/encoder/layer_8/output/LayerNorm/gamma&bert/encoder/layer_8/output/dense/bias(bert/encoder/layer_8/output/dense/kernel4bert/encoder/layer_9/attention/output/LayerNorm/beta5bert/encoder/layer_9/attention/output/LayerNorm/gamma0bert/encoder/layer_9/attention/output/dense/bias2bert/encoder/layer_9/attention/output/dense/kernel,bert/encoder/layer_9/attention/self/key/bias.bert/encoder/layer_9/attention/self/key/kernel.bert/encoder/layer_9/attention/self/query/bias0bert/encoder/layer_9/attention/self/query/kernel.bert/encoder/layer_9/attention/self/value/bias0bert/encoder/layer_9/attention/self/value/kernel,bert/encoder/layer_9/intermediate/dense/bias.bert/encoder/layer_9/intermediate/dense/kernel*bert/encoder/layer_9/output/LayerNorm/beta+bert/encoder/layer_9/output/LayerNorm/gamma&bert/encoder/layer_9/output/dense/bias(bert/encoder/layer_9/output/dense/kernelbert/pooler/dense/biasbert/pooler/dense/kerneloutput_biasoutput_weights*Ú
dtypesÏ
Ì2É
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_class
loc:@save/Const*
_output_shapes
: *
T0
ìJ
save/RestoreV2/tensor_namesConst*J
valueJBJÉBbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelB5bert/encoder/layer_10/attention/output/LayerNorm/betaB6bert/encoder/layer_10/attention/output/LayerNorm/gammaB1bert/encoder/layer_10/attention/output/dense/biasB3bert/encoder/layer_10/attention/output/dense/kernelB-bert/encoder/layer_10/attention/self/key/biasB/bert/encoder/layer_10/attention/self/key/kernelB/bert/encoder/layer_10/attention/self/query/biasB1bert/encoder/layer_10/attention/self/query/kernelB/bert/encoder/layer_10/attention/self/value/biasB1bert/encoder/layer_10/attention/self/value/kernelB-bert/encoder/layer_10/intermediate/dense/biasB/bert/encoder/layer_10/intermediate/dense/kernelB+bert/encoder/layer_10/output/LayerNorm/betaB,bert/encoder/layer_10/output/LayerNorm/gammaB'bert/encoder/layer_10/output/dense/biasB)bert/encoder/layer_10/output/dense/kernelB5bert/encoder/layer_11/attention/output/LayerNorm/betaB6bert/encoder/layer_11/attention/output/LayerNorm/gammaB1bert/encoder/layer_11/attention/output/dense/biasB3bert/encoder/layer_11/attention/output/dense/kernelB-bert/encoder/layer_11/attention/self/key/biasB/bert/encoder/layer_11/attention/self/key/kernelB/bert/encoder/layer_11/attention/self/query/biasB1bert/encoder/layer_11/attention/self/query/kernelB/bert/encoder/layer_11/attention/self/value/biasB1bert/encoder/layer_11/attention/self/value/kernelB-bert/encoder/layer_11/intermediate/dense/biasB/bert/encoder/layer_11/intermediate/dense/kernelB+bert/encoder/layer_11/output/LayerNorm/betaB,bert/encoder/layer_11/output/LayerNorm/gammaB'bert/encoder/layer_11/output/dense/biasB)bert/encoder/layer_11/output/dense/kernelB4bert/encoder/layer_2/attention/output/LayerNorm/betaB5bert/encoder/layer_2/attention/output/LayerNorm/gammaB0bert/encoder/layer_2/attention/output/dense/biasB2bert/encoder/layer_2/attention/output/dense/kernelB,bert/encoder/layer_2/attention/self/key/biasB.bert/encoder/layer_2/attention/self/key/kernelB.bert/encoder/layer_2/attention/self/query/biasB0bert/encoder/layer_2/attention/self/query/kernelB.bert/encoder/layer_2/attention/self/value/biasB0bert/encoder/layer_2/attention/self/value/kernelB,bert/encoder/layer_2/intermediate/dense/biasB.bert/encoder/layer_2/intermediate/dense/kernelB*bert/encoder/layer_2/output/LayerNorm/betaB+bert/encoder/layer_2/output/LayerNorm/gammaB&bert/encoder/layer_2/output/dense/biasB(bert/encoder/layer_2/output/dense/kernelB4bert/encoder/layer_3/attention/output/LayerNorm/betaB5bert/encoder/layer_3/attention/output/LayerNorm/gammaB0bert/encoder/layer_3/attention/output/dense/biasB2bert/encoder/layer_3/attention/output/dense/kernelB,bert/encoder/layer_3/attention/self/key/biasB.bert/encoder/layer_3/attention/self/key/kernelB.bert/encoder/layer_3/attention/self/query/biasB0bert/encoder/layer_3/attention/self/query/kernelB.bert/encoder/layer_3/attention/self/value/biasB0bert/encoder/layer_3/attention/self/value/kernelB,bert/encoder/layer_3/intermediate/dense/biasB.bert/encoder/layer_3/intermediate/dense/kernelB*bert/encoder/layer_3/output/LayerNorm/betaB+bert/encoder/layer_3/output/LayerNorm/gammaB&bert/encoder/layer_3/output/dense/biasB(bert/encoder/layer_3/output/dense/kernelB4bert/encoder/layer_4/attention/output/LayerNorm/betaB5bert/encoder/layer_4/attention/output/LayerNorm/gammaB0bert/encoder/layer_4/attention/output/dense/biasB2bert/encoder/layer_4/attention/output/dense/kernelB,bert/encoder/layer_4/attention/self/key/biasB.bert/encoder/layer_4/attention/self/key/kernelB.bert/encoder/layer_4/attention/self/query/biasB0bert/encoder/layer_4/attention/self/query/kernelB.bert/encoder/layer_4/attention/self/value/biasB0bert/encoder/layer_4/attention/self/value/kernelB,bert/encoder/layer_4/intermediate/dense/biasB.bert/encoder/layer_4/intermediate/dense/kernelB*bert/encoder/layer_4/output/LayerNorm/betaB+bert/encoder/layer_4/output/LayerNorm/gammaB&bert/encoder/layer_4/output/dense/biasB(bert/encoder/layer_4/output/dense/kernelB4bert/encoder/layer_5/attention/output/LayerNorm/betaB5bert/encoder/layer_5/attention/output/LayerNorm/gammaB0bert/encoder/layer_5/attention/output/dense/biasB2bert/encoder/layer_5/attention/output/dense/kernelB,bert/encoder/layer_5/attention/self/key/biasB.bert/encoder/layer_5/attention/self/key/kernelB.bert/encoder/layer_5/attention/self/query/biasB0bert/encoder/layer_5/attention/self/query/kernelB.bert/encoder/layer_5/attention/self/value/biasB0bert/encoder/layer_5/attention/self/value/kernelB,bert/encoder/layer_5/intermediate/dense/biasB.bert/encoder/layer_5/intermediate/dense/kernelB*bert/encoder/layer_5/output/LayerNorm/betaB+bert/encoder/layer_5/output/LayerNorm/gammaB&bert/encoder/layer_5/output/dense/biasB(bert/encoder/layer_5/output/dense/kernelB4bert/encoder/layer_6/attention/output/LayerNorm/betaB5bert/encoder/layer_6/attention/output/LayerNorm/gammaB0bert/encoder/layer_6/attention/output/dense/biasB2bert/encoder/layer_6/attention/output/dense/kernelB,bert/encoder/layer_6/attention/self/key/biasB.bert/encoder/layer_6/attention/self/key/kernelB.bert/encoder/layer_6/attention/self/query/biasB0bert/encoder/layer_6/attention/self/query/kernelB.bert/encoder/layer_6/attention/self/value/biasB0bert/encoder/layer_6/attention/self/value/kernelB,bert/encoder/layer_6/intermediate/dense/biasB.bert/encoder/layer_6/intermediate/dense/kernelB*bert/encoder/layer_6/output/LayerNorm/betaB+bert/encoder/layer_6/output/LayerNorm/gammaB&bert/encoder/layer_6/output/dense/biasB(bert/encoder/layer_6/output/dense/kernelB4bert/encoder/layer_7/attention/output/LayerNorm/betaB5bert/encoder/layer_7/attention/output/LayerNorm/gammaB0bert/encoder/layer_7/attention/output/dense/biasB2bert/encoder/layer_7/attention/output/dense/kernelB,bert/encoder/layer_7/attention/self/key/biasB.bert/encoder/layer_7/attention/self/key/kernelB.bert/encoder/layer_7/attention/self/query/biasB0bert/encoder/layer_7/attention/self/query/kernelB.bert/encoder/layer_7/attention/self/value/biasB0bert/encoder/layer_7/attention/self/value/kernelB,bert/encoder/layer_7/intermediate/dense/biasB.bert/encoder/layer_7/intermediate/dense/kernelB*bert/encoder/layer_7/output/LayerNorm/betaB+bert/encoder/layer_7/output/LayerNorm/gammaB&bert/encoder/layer_7/output/dense/biasB(bert/encoder/layer_7/output/dense/kernelB4bert/encoder/layer_8/attention/output/LayerNorm/betaB5bert/encoder/layer_8/attention/output/LayerNorm/gammaB0bert/encoder/layer_8/attention/output/dense/biasB2bert/encoder/layer_8/attention/output/dense/kernelB,bert/encoder/layer_8/attention/self/key/biasB.bert/encoder/layer_8/attention/self/key/kernelB.bert/encoder/layer_8/attention/self/query/biasB0bert/encoder/layer_8/attention/self/query/kernelB.bert/encoder/layer_8/attention/self/value/biasB0bert/encoder/layer_8/attention/self/value/kernelB,bert/encoder/layer_8/intermediate/dense/biasB.bert/encoder/layer_8/intermediate/dense/kernelB*bert/encoder/layer_8/output/LayerNorm/betaB+bert/encoder/layer_8/output/LayerNorm/gammaB&bert/encoder/layer_8/output/dense/biasB(bert/encoder/layer_8/output/dense/kernelB4bert/encoder/layer_9/attention/output/LayerNorm/betaB5bert/encoder/layer_9/attention/output/LayerNorm/gammaB0bert/encoder/layer_9/attention/output/dense/biasB2bert/encoder/layer_9/attention/output/dense/kernelB,bert/encoder/layer_9/attention/self/key/biasB.bert/encoder/layer_9/attention/self/key/kernelB.bert/encoder/layer_9/attention/self/query/biasB0bert/encoder/layer_9/attention/self/query/kernelB.bert/encoder/layer_9/attention/self/value/biasB0bert/encoder/layer_9/attention/self/value/kernelB,bert/encoder/layer_9/intermediate/dense/biasB.bert/encoder/layer_9/intermediate/dense/kernelB*bert/encoder/layer_9/output/LayerNorm/betaB+bert/encoder/layer_9/output/LayerNorm/gammaB&bert/encoder/layer_9/output/dense/biasB(bert/encoder/layer_9/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBoutput_biasBoutput_weights*
dtype0*
_output_shapes	
:É
ý
save/RestoreV2/shape_and_slicesConst*
_output_shapes	
:É*¨
valueBÉB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
ÿ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Ú
dtypesÏ
Ì2É
Ç
save/AssignAssignbert/embeddings/LayerNorm/betasave/RestoreV2*
use_locking(*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
Í
save/Assign_1Assignbert/embeddings/LayerNorm/gammasave/RestoreV2:1*
_output_shapes	
:*
use_locking(*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
validate_shape(
Ú
save/Assign_2Assign#bert/embeddings/position_embeddingssave/RestoreV2:2* 
_output_shapes
:
*
use_locking(*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings*
validate_shape(
Ý
save/Assign_3Assign%bert/embeddings/token_type_embeddingssave/RestoreV2:3*
_output_shapes
:	*
use_locking(*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
validate_shape(
Ó
save/Assign_4Assignbert/embeddings/word_embeddingssave/RestoreV2:4*
use_locking(*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*
validate_shape(*!
_output_shapes
:ºî
÷
save/Assign_5Assign4bert/encoder/layer_0/attention/output/LayerNorm/betasave/RestoreV2:5*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ù
save/Assign_6Assign5bert/encoder/layer_0/attention/output/LayerNorm/gammasave/RestoreV2:6*
T0*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ï
save/Assign_7Assign0bert/encoder/layer_0/attention/output/dense/biassave/RestoreV2:7*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
ø
save/Assign_8Assign2bert/encoder/layer_0/attention/output/dense/kernelsave/RestoreV2:8*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ç
save/Assign_9Assign,bert/encoder/layer_0/attention/self/key/biassave/RestoreV2:9*
T0*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ò
save/Assign_10Assign.bert/encoder/layer_0/attention/self/key/kernelsave/RestoreV2:10*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
í
save/Assign_11Assign.bert/encoder/layer_0/attention/self/query/biassave/RestoreV2:11*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ö
save/Assign_12Assign0bert/encoder/layer_0/attention/self/query/kernelsave/RestoreV2:12*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

í
save/Assign_13Assign.bert/encoder/layer_0/attention/self/value/biassave/RestoreV2:13*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
ö
save/Assign_14Assign0bert/encoder/layer_0/attention/self/value/kernelsave/RestoreV2:14*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
é
save/Assign_15Assign,bert/encoder/layer_0/intermediate/dense/biassave/RestoreV2:15*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ò
save/Assign_16Assign.bert/encoder/layer_0/intermediate/dense/kernelsave/RestoreV2:16*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

å
save/Assign_17Assign*bert/encoder/layer_0/output/LayerNorm/betasave/RestoreV2:17*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ç
save/Assign_18Assign+bert/encoder/layer_0/output/LayerNorm/gammasave/RestoreV2:18*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
validate_shape(
Ý
save/Assign_19Assign&bert/encoder/layer_0/output/dense/biassave/RestoreV2:19*
_output_shapes	
:*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
validate_shape(
æ
save/Assign_20Assign(bert/encoder/layer_0/output/dense/kernelsave/RestoreV2:20* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
validate_shape(
ù
save/Assign_21Assign4bert/encoder/layer_1/attention/output/LayerNorm/betasave/RestoreV2:21*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
û
save/Assign_22Assign5bert/encoder/layer_1/attention/output/LayerNorm/gammasave/RestoreV2:22*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ñ
save/Assign_23Assign0bert/encoder/layer_1/attention/output/dense/biassave/RestoreV2:23*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ú
save/Assign_24Assign2bert/encoder/layer_1/attention/output/dense/kernelsave/RestoreV2:24*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
é
save/Assign_25Assign,bert/encoder/layer_1/attention/self/key/biassave/RestoreV2:25*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
validate_shape(
ò
save/Assign_26Assign.bert/encoder/layer_1/attention/self/key/kernelsave/RestoreV2:26*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel
í
save/Assign_27Assign.bert/encoder/layer_1/attention/self/query/biassave/RestoreV2:27*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
save/Assign_28Assign0bert/encoder/layer_1/attention/self/query/kernelsave/RestoreV2:28*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

í
save/Assign_29Assign.bert/encoder/layer_1/attention/self/value/biassave/RestoreV2:29*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias
ö
save/Assign_30Assign0bert/encoder/layer_1/attention/self/value/kernelsave/RestoreV2:30*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

é
save/Assign_31Assign,bert/encoder/layer_1/intermediate/dense/biassave/RestoreV2:31*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ò
save/Assign_32Assign.bert/encoder/layer_1/intermediate/dense/kernelsave/RestoreV2:32*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel
å
save/Assign_33Assign*bert/encoder/layer_1/output/LayerNorm/betasave/RestoreV2:33*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ç
save/Assign_34Assign+bert/encoder/layer_1/output/LayerNorm/gammasave/RestoreV2:34*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma
Ý
save/Assign_35Assign&bert/encoder/layer_1/output/dense/biassave/RestoreV2:35*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias
æ
save/Assign_36Assign(bert/encoder/layer_1/output/dense/kernelsave/RestoreV2:36* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
validate_shape(
û
save/Assign_37Assign5bert/encoder/layer_10/attention/output/LayerNorm/betasave/RestoreV2:37*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta
ý
save/Assign_38Assign6bert/encoder/layer_10/attention/output/LayerNorm/gammasave/RestoreV2:38*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ó
save/Assign_39Assign1bert/encoder/layer_10/attention/output/dense/biassave/RestoreV2:39*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
ü
save/Assign_40Assign3bert/encoder/layer_10/attention/output/dense/kernelsave/RestoreV2:40*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel
ë
save/Assign_41Assign-bert/encoder/layer_10/attention/self/key/biassave/RestoreV2:41*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_42Assign/bert/encoder/layer_10/attention/self/key/kernelsave/RestoreV2:42*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_43Assign/bert/encoder/layer_10/attention/self/query/biassave/RestoreV2:43*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save/Assign_44Assign1bert/encoder/layer_10/attention/self/query/kernelsave/RestoreV2:44*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_45Assign/bert/encoder/layer_10/attention/self/value/biassave/RestoreV2:45*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save/Assign_46Assign1bert/encoder/layer_10/attention/self/value/kernelsave/RestoreV2:46*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel
ë
save/Assign_47Assign-bert/encoder/layer_10/intermediate/dense/biassave/RestoreV2:47*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_48Assign/bert/encoder/layer_10/intermediate/dense/kernelsave/RestoreV2:48*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

ç
save/Assign_49Assign+bert/encoder/layer_10/output/LayerNorm/betasave/RestoreV2:49*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
é
save/Assign_50Assign,bert/encoder/layer_10/output/LayerNorm/gammasave/RestoreV2:50*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ß
save/Assign_51Assign'bert/encoder/layer_10/output/dense/biassave/RestoreV2:51*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
validate_shape(*
_output_shapes	
:
è
save/Assign_52Assign)bert/encoder/layer_10/output/dense/kernelsave/RestoreV2:52*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
û
save/Assign_53Assign5bert/encoder/layer_11/attention/output/LayerNorm/betasave/RestoreV2:53*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta
ý
save/Assign_54Assign6bert/encoder/layer_11/attention/output/LayerNorm/gammasave/RestoreV2:54*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ó
save/Assign_55Assign1bert/encoder/layer_11/attention/output/dense/biassave/RestoreV2:55*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias
ü
save/Assign_56Assign3bert/encoder/layer_11/attention/output/dense/kernelsave/RestoreV2:56* 
_output_shapes
:
*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
validate_shape(
ë
save/Assign_57Assign-bert/encoder/layer_11/attention/self/key/biassave/RestoreV2:57*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias*
validate_shape(
ô
save/Assign_58Assign/bert/encoder/layer_11/attention/self/key/kernelsave/RestoreV2:58*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_59Assign/bert/encoder/layer_11/attention/self/query/biassave/RestoreV2:59*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save/Assign_60Assign1bert/encoder/layer_11/attention/self/query/kernelsave/RestoreV2:60*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel
ï
save/Assign_61Assign/bert/encoder/layer_11/attention/self/value/biassave/RestoreV2:61*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save/Assign_62Assign1bert/encoder/layer_11/attention/self/value/kernelsave/RestoreV2:62*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ë
save/Assign_63Assign-bert/encoder/layer_11/intermediate/dense/biassave/RestoreV2:63*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_64Assign/bert/encoder/layer_11/intermediate/dense/kernelsave/RestoreV2:64*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel
ç
save/Assign_65Assign+bert/encoder/layer_11/output/LayerNorm/betasave/RestoreV2:65*
T0*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
é
save/Assign_66Assign,bert/encoder/layer_11/output/LayerNorm/gammasave/RestoreV2:66*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ß
save/Assign_67Assign'bert/encoder/layer_11/output/dense/biassave/RestoreV2:67*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
validate_shape(*
_output_shapes	
:
è
save/Assign_68Assign)bert/encoder/layer_11/output/dense/kernelsave/RestoreV2:68*
use_locking(*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ù
save/Assign_69Assign4bert/encoder/layer_2/attention/output/LayerNorm/betasave/RestoreV2:69*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
û
save/Assign_70Assign5bert/encoder/layer_2/attention/output/LayerNorm/gammasave/RestoreV2:70*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
validate_shape(
ñ
save/Assign_71Assign0bert/encoder/layer_2/attention/output/dense/biassave/RestoreV2:71*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
validate_shape(
ú
save/Assign_72Assign2bert/encoder/layer_2/attention/output/dense/kernelsave/RestoreV2:72*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel
é
save/Assign_73Assign,bert/encoder/layer_2/attention/self/key/biassave/RestoreV2:73*
T0*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ò
save/Assign_74Assign.bert/encoder/layer_2/attention/self/key/kernelsave/RestoreV2:74*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

í
save/Assign_75Assign.bert/encoder/layer_2/attention/self/query/biassave/RestoreV2:75*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
validate_shape(
ö
save/Assign_76Assign0bert/encoder/layer_2/attention/self/query/kernelsave/RestoreV2:76*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel
í
save/Assign_77Assign.bert/encoder/layer_2/attention/self/value/biassave/RestoreV2:77*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ö
save/Assign_78Assign0bert/encoder/layer_2/attention/self/value/kernelsave/RestoreV2:78*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
é
save/Assign_79Assign,bert/encoder/layer_2/intermediate/dense/biassave/RestoreV2:79*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
validate_shape(
ò
save/Assign_80Assign.bert/encoder/layer_2/intermediate/dense/kernelsave/RestoreV2:80*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

å
save/Assign_81Assign*bert/encoder/layer_2/output/LayerNorm/betasave/RestoreV2:81*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ç
save/Assign_82Assign+bert/encoder/layer_2/output/LayerNorm/gammasave/RestoreV2:82*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
Ý
save/Assign_83Assign&bert/encoder/layer_2/output/dense/biassave/RestoreV2:83*
_output_shapes	
:*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
validate_shape(
æ
save/Assign_84Assign(bert/encoder/layer_2/output/dense/kernelsave/RestoreV2:84*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ù
save/Assign_85Assign4bert/encoder/layer_3/attention/output/LayerNorm/betasave/RestoreV2:85*
T0*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
û
save/Assign_86Assign5bert/encoder/layer_3/attention/output/LayerNorm/gammasave/RestoreV2:86*
T0*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ñ
save/Assign_87Assign0bert/encoder/layer_3/attention/output/dense/biassave/RestoreV2:87*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias
ú
save/Assign_88Assign2bert/encoder/layer_3/attention/output/dense/kernelsave/RestoreV2:88*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

é
save/Assign_89Assign,bert/encoder/layer_3/attention/self/key/biassave/RestoreV2:89*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ò
save/Assign_90Assign.bert/encoder/layer_3/attention/self/key/kernelsave/RestoreV2:90*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel
í
save/Assign_91Assign.bert/encoder/layer_3/attention/self/query/biassave/RestoreV2:91*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ö
save/Assign_92Assign0bert/encoder/layer_3/attention/self/query/kernelsave/RestoreV2:92*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
í
save/Assign_93Assign.bert/encoder/layer_3/attention/self/value/biassave/RestoreV2:93*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
ö
save/Assign_94Assign0bert/encoder/layer_3/attention/self/value/kernelsave/RestoreV2:94*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

é
save/Assign_95Assign,bert/encoder/layer_3/intermediate/dense/biassave/RestoreV2:95*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
validate_shape(
ò
save/Assign_96Assign.bert/encoder/layer_3/intermediate/dense/kernelsave/RestoreV2:96*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

å
save/Assign_97Assign*bert/encoder/layer_3/output/LayerNorm/betasave/RestoreV2:97*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ç
save/Assign_98Assign+bert/encoder/layer_3/output/LayerNorm/gammasave/RestoreV2:98*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
validate_shape(
Ý
save/Assign_99Assign&bert/encoder/layer_3/output/dense/biassave/RestoreV2:99*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
validate_shape(*
_output_shapes	
:
è
save/Assign_100Assign(bert/encoder/layer_3/output/dense/kernelsave/RestoreV2:100*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
û
save/Assign_101Assign4bert/encoder/layer_4/attention/output/LayerNorm/betasave/RestoreV2:101*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ý
save/Assign_102Assign5bert/encoder/layer_4/attention/output/LayerNorm/gammasave/RestoreV2:102*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ó
save/Assign_103Assign0bert/encoder/layer_4/attention/output/dense/biassave/RestoreV2:103*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
ü
save/Assign_104Assign2bert/encoder/layer_4/attention/output/dense/kernelsave/RestoreV2:104*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ë
save/Assign_105Assign,bert/encoder/layer_4/attention/self/key/biassave/RestoreV2:105*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_106Assign.bert/encoder/layer_4/attention/self/key/kernelsave/RestoreV2:106*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_107Assign.bert/encoder/layer_4/attention/self/query/biassave/RestoreV2:107*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save/Assign_108Assign0bert/encoder/layer_4/attention/self/query/kernelsave/RestoreV2:108*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_109Assign.bert/encoder/layer_4/attention/self/value/biassave/RestoreV2:109*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
validate_shape(
ø
save/Assign_110Assign0bert/encoder/layer_4/attention/self/value/kernelsave/RestoreV2:110*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ë
save/Assign_111Assign,bert/encoder/layer_4/intermediate/dense/biassave/RestoreV2:111*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_112Assign.bert/encoder/layer_4/intermediate/dense/kernelsave/RestoreV2:112*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

ç
save/Assign_113Assign*bert/encoder/layer_4/output/LayerNorm/betasave/RestoreV2:113*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
validate_shape(
é
save/Assign_114Assign+bert/encoder/layer_4/output/LayerNorm/gammasave/RestoreV2:114*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ß
save/Assign_115Assign&bert/encoder/layer_4/output/dense/biassave/RestoreV2:115*
T0*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
è
save/Assign_116Assign(bert/encoder/layer_4/output/dense/kernelsave/RestoreV2:116*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel
û
save/Assign_117Assign4bert/encoder/layer_5/attention/output/LayerNorm/betasave/RestoreV2:117*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ý
save/Assign_118Assign5bert/encoder/layer_5/attention/output/LayerNorm/gammasave/RestoreV2:118*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ó
save/Assign_119Assign0bert/encoder/layer_5/attention/output/dense/biassave/RestoreV2:119*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ü
save/Assign_120Assign2bert/encoder/layer_5/attention/output/dense/kernelsave/RestoreV2:120*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ë
save/Assign_121Assign,bert/encoder/layer_5/attention/self/key/biassave/RestoreV2:121*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_122Assign.bert/encoder/layer_5/attention/self/key/kernelsave/RestoreV2:122*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_123Assign.bert/encoder/layer_5/attention/self/query/biassave/RestoreV2:123*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
validate_shape(
ø
save/Assign_124Assign0bert/encoder/layer_5/attention/self/query/kernelsave/RestoreV2:124*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_125Assign.bert/encoder/layer_5/attention/self/value/biassave/RestoreV2:125*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save/Assign_126Assign0bert/encoder/layer_5/attention/self/value/kernelsave/RestoreV2:126*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ë
save/Assign_127Assign,bert/encoder/layer_5/intermediate/dense/biassave/RestoreV2:127*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
validate_shape(
ô
save/Assign_128Assign.bert/encoder/layer_5/intermediate/dense/kernelsave/RestoreV2:128*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

ç
save/Assign_129Assign*bert/encoder/layer_5/output/LayerNorm/betasave/RestoreV2:129*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
é
save/Assign_130Assign+bert/encoder/layer_5/output/LayerNorm/gammasave/RestoreV2:130*
T0*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ß
save/Assign_131Assign&bert/encoder/layer_5/output/dense/biassave/RestoreV2:131*
T0*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
è
save/Assign_132Assign(bert/encoder/layer_5/output/dense/kernelsave/RestoreV2:132*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel*
validate_shape(* 
_output_shapes
:

û
save/Assign_133Assign4bert/encoder/layer_6/attention/output/LayerNorm/betasave/RestoreV2:133*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ý
save/Assign_134Assign5bert/encoder/layer_6/attention/output/LayerNorm/gammasave/RestoreV2:134*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ó
save/Assign_135Assign0bert/encoder/layer_6/attention/output/dense/biassave/RestoreV2:135*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
ü
save/Assign_136Assign2bert/encoder/layer_6/attention/output/dense/kernelsave/RestoreV2:136*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel
ë
save/Assign_137Assign,bert/encoder/layer_6/attention/self/key/biassave/RestoreV2:137*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_138Assign.bert/encoder/layer_6/attention/self/key/kernelsave/RestoreV2:138* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
validate_shape(
ï
save/Assign_139Assign.bert/encoder/layer_6/attention/self/query/biassave/RestoreV2:139*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save/Assign_140Assign0bert/encoder/layer_6/attention/self/query/kernelsave/RestoreV2:140*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_141Assign.bert/encoder/layer_6/attention/self/value/biassave/RestoreV2:141*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias
ø
save/Assign_142Assign0bert/encoder/layer_6/attention/self/value/kernelsave/RestoreV2:142*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel
ë
save/Assign_143Assign,bert/encoder/layer_6/intermediate/dense/biassave/RestoreV2:143*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_144Assign.bert/encoder/layer_6/intermediate/dense/kernelsave/RestoreV2:144* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
validate_shape(
ç
save/Assign_145Assign*bert/encoder/layer_6/output/LayerNorm/betasave/RestoreV2:145*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
validate_shape(
é
save/Assign_146Assign+bert/encoder/layer_6/output/LayerNorm/gammasave/RestoreV2:146*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma
ß
save/Assign_147Assign&bert/encoder/layer_6/output/dense/biassave/RestoreV2:147*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
è
save/Assign_148Assign(bert/encoder/layer_6/output/dense/kernelsave/RestoreV2:148*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel
û
save/Assign_149Assign4bert/encoder/layer_7/attention/output/LayerNorm/betasave/RestoreV2:149*
_output_shapes	
:*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
validate_shape(
ý
save/Assign_150Assign5bert/encoder/layer_7/attention/output/LayerNorm/gammasave/RestoreV2:150*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ó
save/Assign_151Assign0bert/encoder/layer_7/attention/output/dense/biassave/RestoreV2:151*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
ü
save/Assign_152Assign2bert/encoder/layer_7/attention/output/dense/kernelsave/RestoreV2:152*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ë
save/Assign_153Assign,bert/encoder/layer_7/attention/self/key/biassave/RestoreV2:153*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_154Assign.bert/encoder/layer_7/attention/self/key/kernelsave/RestoreV2:154*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ï
save/Assign_155Assign.bert/encoder/layer_7/attention/self/query/biassave/RestoreV2:155*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ø
save/Assign_156Assign0bert/encoder/layer_7/attention/self/query/kernelsave/RestoreV2:156*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_157Assign.bert/encoder/layer_7/attention/self/value/biassave/RestoreV2:157*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save/Assign_158Assign0bert/encoder/layer_7/attention/self/value/kernelsave/RestoreV2:158*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ë
save/Assign_159Assign,bert/encoder/layer_7/intermediate/dense/biassave/RestoreV2:159*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_160Assign.bert/encoder/layer_7/intermediate/dense/kernelsave/RestoreV2:160*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ç
save/Assign_161Assign*bert/encoder/layer_7/output/LayerNorm/betasave/RestoreV2:161*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
é
save/Assign_162Assign+bert/encoder/layer_7/output/LayerNorm/gammasave/RestoreV2:162*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ß
save/Assign_163Assign&bert/encoder/layer_7/output/dense/biassave/RestoreV2:163*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
validate_shape(*
_output_shapes	
:
è
save/Assign_164Assign(bert/encoder/layer_7/output/dense/kernelsave/RestoreV2:164*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel
û
save/Assign_165Assign4bert/encoder/layer_8/attention/output/LayerNorm/betasave/RestoreV2:165*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ý
save/Assign_166Assign5bert/encoder/layer_8/attention/output/LayerNorm/gammasave/RestoreV2:166*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ó
save/Assign_167Assign0bert/encoder/layer_8/attention/output/dense/biassave/RestoreV2:167*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
ü
save/Assign_168Assign2bert/encoder/layer_8/attention/output/dense/kernelsave/RestoreV2:168*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ë
save/Assign_169Assign,bert/encoder/layer_8/attention/self/key/biassave/RestoreV2:169*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
save/Assign_170Assign.bert/encoder/layer_8/attention/self/key/kernelsave/RestoreV2:170*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_171Assign.bert/encoder/layer_8/attention/self/query/biassave/RestoreV2:171*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ø
save/Assign_172Assign0bert/encoder/layer_8/attention/self/query/kernelsave/RestoreV2:172*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ï
save/Assign_173Assign.bert/encoder/layer_8/attention/self/value/biassave/RestoreV2:173*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias*
validate_shape(
ø
save/Assign_174Assign0bert/encoder/layer_8/attention/self/value/kernelsave/RestoreV2:174*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ë
save/Assign_175Assign,bert/encoder/layer_8/intermediate/dense/biassave/RestoreV2:175*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ô
save/Assign_176Assign.bert/encoder/layer_8/intermediate/dense/kernelsave/RestoreV2:176*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ç
save/Assign_177Assign*bert/encoder/layer_8/output/LayerNorm/betasave/RestoreV2:177*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
é
save/Assign_178Assign+bert/encoder/layer_8/output/LayerNorm/gammasave/RestoreV2:178*
T0*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ß
save/Assign_179Assign&bert/encoder/layer_8/output/dense/biassave/RestoreV2:179*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
validate_shape(*
_output_shapes	
:
è
save/Assign_180Assign(bert/encoder/layer_8/output/dense/kernelsave/RestoreV2:180*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
validate_shape(* 
_output_shapes
:

û
save/Assign_181Assign4bert/encoder/layer_9/attention/output/LayerNorm/betasave/RestoreV2:181*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ý
save/Assign_182Assign5bert/encoder/layer_9/attention/output/LayerNorm/gammasave/RestoreV2:182*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ó
save/Assign_183Assign0bert/encoder/layer_9/attention/output/dense/biassave/RestoreV2:183*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ü
save/Assign_184Assign2bert/encoder/layer_9/attention/output/dense/kernelsave/RestoreV2:184*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ë
save/Assign_185Assign,bert/encoder/layer_9/attention/self/key/biassave/RestoreV2:185*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ô
save/Assign_186Assign.bert/encoder/layer_9/attention/self/key/kernelsave/RestoreV2:186*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ï
save/Assign_187Assign.bert/encoder/layer_9/attention/self/query/biassave/RestoreV2:187*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ø
save/Assign_188Assign0bert/encoder/layer_9/attention/self/query/kernelsave/RestoreV2:188*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ï
save/Assign_189Assign.bert/encoder/layer_9/attention/self/value/biassave/RestoreV2:189*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save/Assign_190Assign0bert/encoder/layer_9/attention/self/value/kernelsave/RestoreV2:190*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ë
save/Assign_191Assign,bert/encoder/layer_9/intermediate/dense/biassave/RestoreV2:191*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias
ô
save/Assign_192Assign.bert/encoder/layer_9/intermediate/dense/kernelsave/RestoreV2:192*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

ç
save/Assign_193Assign*bert/encoder/layer_9/output/LayerNorm/betasave/RestoreV2:193*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
é
save/Assign_194Assign+bert/encoder/layer_9/output/LayerNorm/gammasave/RestoreV2:194*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ß
save/Assign_195Assign&bert/encoder/layer_9/output/dense/biassave/RestoreV2:195*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
validate_shape(*
_output_shapes	
:
è
save/Assign_196Assign(bert/encoder/layer_9/output/dense/kernelsave/RestoreV2:196*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
¿
save/Assign_197Assignbert/pooler/dense/biassave/RestoreV2:197*
_output_shapes	
:*
use_locking(*
T0*)
_class
loc:@bert/pooler/dense/bias*
validate_shape(
È
save/Assign_198Assignbert/pooler/dense/kernelsave/RestoreV2:198*+
_class!
loc:@bert/pooler/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
¨
save/Assign_199Assignoutput_biassave/RestoreV2:199*
_class
loc:@output_bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
³
save/Assign_200Assignoutput_weightssave/RestoreV2:200*
T0*!
_class
loc:@output_weights*
validate_shape(*
_output_shapes
:	*
use_locking(
Ê
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_13^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_14^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_147^save/Assign_148^save/Assign_149^save/Assign_15^save/Assign_150^save/Assign_151^save/Assign_152^save/Assign_153^save/Assign_154^save/Assign_155^save/Assign_156^save/Assign_157^save/Assign_158^save/Assign_159^save/Assign_16^save/Assign_160^save/Assign_161^save/Assign_162^save/Assign_163^save/Assign_164^save/Assign_165^save/Assign_166^save/Assign_167^save/Assign_168^save/Assign_169^save/Assign_17^save/Assign_170^save/Assign_171^save/Assign_172^save/Assign_173^save/Assign_174^save/Assign_175^save/Assign_176^save/Assign_177^save/Assign_178^save/Assign_179^save/Assign_18^save/Assign_180^save/Assign_181^save/Assign_182^save/Assign_183^save/Assign_184^save/Assign_185^save/Assign_186^save/Assign_187^save/Assign_188^save/Assign_189^save/Assign_19^save/Assign_190^save/Assign_191^save/Assign_192^save/Assign_193^save/Assign_194^save/Assign_195^save/Assign_196^save/Assign_197^save/Assign_198^save/Assign_199^save/Assign_2^save/Assign_20^save/Assign_200^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
[
save_1/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
_output_shapes
: *
shape: *
dtype0
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 

save_1/StringJoin/inputs_1Const*
_output_shapes
: *<
value3B1 B+_temp_c9a7ddc30f86424c9682244197e75f87/part*
dtype0
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_1/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
ëJ
save_1/SaveV2/tensor_namesConst*J
valueJBJÉBbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelB5bert/encoder/layer_10/attention/output/LayerNorm/betaB6bert/encoder/layer_10/attention/output/LayerNorm/gammaB1bert/encoder/layer_10/attention/output/dense/biasB3bert/encoder/layer_10/attention/output/dense/kernelB-bert/encoder/layer_10/attention/self/key/biasB/bert/encoder/layer_10/attention/self/key/kernelB/bert/encoder/layer_10/attention/self/query/biasB1bert/encoder/layer_10/attention/self/query/kernelB/bert/encoder/layer_10/attention/self/value/biasB1bert/encoder/layer_10/attention/self/value/kernelB-bert/encoder/layer_10/intermediate/dense/biasB/bert/encoder/layer_10/intermediate/dense/kernelB+bert/encoder/layer_10/output/LayerNorm/betaB,bert/encoder/layer_10/output/LayerNorm/gammaB'bert/encoder/layer_10/output/dense/biasB)bert/encoder/layer_10/output/dense/kernelB5bert/encoder/layer_11/attention/output/LayerNorm/betaB6bert/encoder/layer_11/attention/output/LayerNorm/gammaB1bert/encoder/layer_11/attention/output/dense/biasB3bert/encoder/layer_11/attention/output/dense/kernelB-bert/encoder/layer_11/attention/self/key/biasB/bert/encoder/layer_11/attention/self/key/kernelB/bert/encoder/layer_11/attention/self/query/biasB1bert/encoder/layer_11/attention/self/query/kernelB/bert/encoder/layer_11/attention/self/value/biasB1bert/encoder/layer_11/attention/self/value/kernelB-bert/encoder/layer_11/intermediate/dense/biasB/bert/encoder/layer_11/intermediate/dense/kernelB+bert/encoder/layer_11/output/LayerNorm/betaB,bert/encoder/layer_11/output/LayerNorm/gammaB'bert/encoder/layer_11/output/dense/biasB)bert/encoder/layer_11/output/dense/kernelB4bert/encoder/layer_2/attention/output/LayerNorm/betaB5bert/encoder/layer_2/attention/output/LayerNorm/gammaB0bert/encoder/layer_2/attention/output/dense/biasB2bert/encoder/layer_2/attention/output/dense/kernelB,bert/encoder/layer_2/attention/self/key/biasB.bert/encoder/layer_2/attention/self/key/kernelB.bert/encoder/layer_2/attention/self/query/biasB0bert/encoder/layer_2/attention/self/query/kernelB.bert/encoder/layer_2/attention/self/value/biasB0bert/encoder/layer_2/attention/self/value/kernelB,bert/encoder/layer_2/intermediate/dense/biasB.bert/encoder/layer_2/intermediate/dense/kernelB*bert/encoder/layer_2/output/LayerNorm/betaB+bert/encoder/layer_2/output/LayerNorm/gammaB&bert/encoder/layer_2/output/dense/biasB(bert/encoder/layer_2/output/dense/kernelB4bert/encoder/layer_3/attention/output/LayerNorm/betaB5bert/encoder/layer_3/attention/output/LayerNorm/gammaB0bert/encoder/layer_3/attention/output/dense/biasB2bert/encoder/layer_3/attention/output/dense/kernelB,bert/encoder/layer_3/attention/self/key/biasB.bert/encoder/layer_3/attention/self/key/kernelB.bert/encoder/layer_3/attention/self/query/biasB0bert/encoder/layer_3/attention/self/query/kernelB.bert/encoder/layer_3/attention/self/value/biasB0bert/encoder/layer_3/attention/self/value/kernelB,bert/encoder/layer_3/intermediate/dense/biasB.bert/encoder/layer_3/intermediate/dense/kernelB*bert/encoder/layer_3/output/LayerNorm/betaB+bert/encoder/layer_3/output/LayerNorm/gammaB&bert/encoder/layer_3/output/dense/biasB(bert/encoder/layer_3/output/dense/kernelB4bert/encoder/layer_4/attention/output/LayerNorm/betaB5bert/encoder/layer_4/attention/output/LayerNorm/gammaB0bert/encoder/layer_4/attention/output/dense/biasB2bert/encoder/layer_4/attention/output/dense/kernelB,bert/encoder/layer_4/attention/self/key/biasB.bert/encoder/layer_4/attention/self/key/kernelB.bert/encoder/layer_4/attention/self/query/biasB0bert/encoder/layer_4/attention/self/query/kernelB.bert/encoder/layer_4/attention/self/value/biasB0bert/encoder/layer_4/attention/self/value/kernelB,bert/encoder/layer_4/intermediate/dense/biasB.bert/encoder/layer_4/intermediate/dense/kernelB*bert/encoder/layer_4/output/LayerNorm/betaB+bert/encoder/layer_4/output/LayerNorm/gammaB&bert/encoder/layer_4/output/dense/biasB(bert/encoder/layer_4/output/dense/kernelB4bert/encoder/layer_5/attention/output/LayerNorm/betaB5bert/encoder/layer_5/attention/output/LayerNorm/gammaB0bert/encoder/layer_5/attention/output/dense/biasB2bert/encoder/layer_5/attention/output/dense/kernelB,bert/encoder/layer_5/attention/self/key/biasB.bert/encoder/layer_5/attention/self/key/kernelB.bert/encoder/layer_5/attention/self/query/biasB0bert/encoder/layer_5/attention/self/query/kernelB.bert/encoder/layer_5/attention/self/value/biasB0bert/encoder/layer_5/attention/self/value/kernelB,bert/encoder/layer_5/intermediate/dense/biasB.bert/encoder/layer_5/intermediate/dense/kernelB*bert/encoder/layer_5/output/LayerNorm/betaB+bert/encoder/layer_5/output/LayerNorm/gammaB&bert/encoder/layer_5/output/dense/biasB(bert/encoder/layer_5/output/dense/kernelB4bert/encoder/layer_6/attention/output/LayerNorm/betaB5bert/encoder/layer_6/attention/output/LayerNorm/gammaB0bert/encoder/layer_6/attention/output/dense/biasB2bert/encoder/layer_6/attention/output/dense/kernelB,bert/encoder/layer_6/attention/self/key/biasB.bert/encoder/layer_6/attention/self/key/kernelB.bert/encoder/layer_6/attention/self/query/biasB0bert/encoder/layer_6/attention/self/query/kernelB.bert/encoder/layer_6/attention/self/value/biasB0bert/encoder/layer_6/attention/self/value/kernelB,bert/encoder/layer_6/intermediate/dense/biasB.bert/encoder/layer_6/intermediate/dense/kernelB*bert/encoder/layer_6/output/LayerNorm/betaB+bert/encoder/layer_6/output/LayerNorm/gammaB&bert/encoder/layer_6/output/dense/biasB(bert/encoder/layer_6/output/dense/kernelB4bert/encoder/layer_7/attention/output/LayerNorm/betaB5bert/encoder/layer_7/attention/output/LayerNorm/gammaB0bert/encoder/layer_7/attention/output/dense/biasB2bert/encoder/layer_7/attention/output/dense/kernelB,bert/encoder/layer_7/attention/self/key/biasB.bert/encoder/layer_7/attention/self/key/kernelB.bert/encoder/layer_7/attention/self/query/biasB0bert/encoder/layer_7/attention/self/query/kernelB.bert/encoder/layer_7/attention/self/value/biasB0bert/encoder/layer_7/attention/self/value/kernelB,bert/encoder/layer_7/intermediate/dense/biasB.bert/encoder/layer_7/intermediate/dense/kernelB*bert/encoder/layer_7/output/LayerNorm/betaB+bert/encoder/layer_7/output/LayerNorm/gammaB&bert/encoder/layer_7/output/dense/biasB(bert/encoder/layer_7/output/dense/kernelB4bert/encoder/layer_8/attention/output/LayerNorm/betaB5bert/encoder/layer_8/attention/output/LayerNorm/gammaB0bert/encoder/layer_8/attention/output/dense/biasB2bert/encoder/layer_8/attention/output/dense/kernelB,bert/encoder/layer_8/attention/self/key/biasB.bert/encoder/layer_8/attention/self/key/kernelB.bert/encoder/layer_8/attention/self/query/biasB0bert/encoder/layer_8/attention/self/query/kernelB.bert/encoder/layer_8/attention/self/value/biasB0bert/encoder/layer_8/attention/self/value/kernelB,bert/encoder/layer_8/intermediate/dense/biasB.bert/encoder/layer_8/intermediate/dense/kernelB*bert/encoder/layer_8/output/LayerNorm/betaB+bert/encoder/layer_8/output/LayerNorm/gammaB&bert/encoder/layer_8/output/dense/biasB(bert/encoder/layer_8/output/dense/kernelB4bert/encoder/layer_9/attention/output/LayerNorm/betaB5bert/encoder/layer_9/attention/output/LayerNorm/gammaB0bert/encoder/layer_9/attention/output/dense/biasB2bert/encoder/layer_9/attention/output/dense/kernelB,bert/encoder/layer_9/attention/self/key/biasB.bert/encoder/layer_9/attention/self/key/kernelB.bert/encoder/layer_9/attention/self/query/biasB0bert/encoder/layer_9/attention/self/query/kernelB.bert/encoder/layer_9/attention/self/value/biasB0bert/encoder/layer_9/attention/self/value/kernelB,bert/encoder/layer_9/intermediate/dense/biasB.bert/encoder/layer_9/intermediate/dense/kernelB*bert/encoder/layer_9/output/LayerNorm/betaB+bert/encoder/layer_9/output/LayerNorm/gammaB&bert/encoder/layer_9/output/dense/biasB(bert/encoder/layer_9/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBoutput_biasBoutput_weights*
dtype0*
_output_shapes	
:É
ü
save_1/SaveV2/shape_and_slicesConst*¨
valueBÉB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes	
:É
ÍL
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbert/embeddings/LayerNorm/betabert/embeddings/LayerNorm/gamma#bert/embeddings/position_embeddings%bert/embeddings/token_type_embeddingsbert/embeddings/word_embeddings4bert/encoder/layer_0/attention/output/LayerNorm/beta5bert/encoder/layer_0/attention/output/LayerNorm/gamma0bert/encoder/layer_0/attention/output/dense/bias2bert/encoder/layer_0/attention/output/dense/kernel,bert/encoder/layer_0/attention/self/key/bias.bert/encoder/layer_0/attention/self/key/kernel.bert/encoder/layer_0/attention/self/query/bias0bert/encoder/layer_0/attention/self/query/kernel.bert/encoder/layer_0/attention/self/value/bias0bert/encoder/layer_0/attention/self/value/kernel,bert/encoder/layer_0/intermediate/dense/bias.bert/encoder/layer_0/intermediate/dense/kernel*bert/encoder/layer_0/output/LayerNorm/beta+bert/encoder/layer_0/output/LayerNorm/gamma&bert/encoder/layer_0/output/dense/bias(bert/encoder/layer_0/output/dense/kernel4bert/encoder/layer_1/attention/output/LayerNorm/beta5bert/encoder/layer_1/attention/output/LayerNorm/gamma0bert/encoder/layer_1/attention/output/dense/bias2bert/encoder/layer_1/attention/output/dense/kernel,bert/encoder/layer_1/attention/self/key/bias.bert/encoder/layer_1/attention/self/key/kernel.bert/encoder/layer_1/attention/self/query/bias0bert/encoder/layer_1/attention/self/query/kernel.bert/encoder/layer_1/attention/self/value/bias0bert/encoder/layer_1/attention/self/value/kernel,bert/encoder/layer_1/intermediate/dense/bias.bert/encoder/layer_1/intermediate/dense/kernel*bert/encoder/layer_1/output/LayerNorm/beta+bert/encoder/layer_1/output/LayerNorm/gamma&bert/encoder/layer_1/output/dense/bias(bert/encoder/layer_1/output/dense/kernel5bert/encoder/layer_10/attention/output/LayerNorm/beta6bert/encoder/layer_10/attention/output/LayerNorm/gamma1bert/encoder/layer_10/attention/output/dense/bias3bert/encoder/layer_10/attention/output/dense/kernel-bert/encoder/layer_10/attention/self/key/bias/bert/encoder/layer_10/attention/self/key/kernel/bert/encoder/layer_10/attention/self/query/bias1bert/encoder/layer_10/attention/self/query/kernel/bert/encoder/layer_10/attention/self/value/bias1bert/encoder/layer_10/attention/self/value/kernel-bert/encoder/layer_10/intermediate/dense/bias/bert/encoder/layer_10/intermediate/dense/kernel+bert/encoder/layer_10/output/LayerNorm/beta,bert/encoder/layer_10/output/LayerNorm/gamma'bert/encoder/layer_10/output/dense/bias)bert/encoder/layer_10/output/dense/kernel5bert/encoder/layer_11/attention/output/LayerNorm/beta6bert/encoder/layer_11/attention/output/LayerNorm/gamma1bert/encoder/layer_11/attention/output/dense/bias3bert/encoder/layer_11/attention/output/dense/kernel-bert/encoder/layer_11/attention/self/key/bias/bert/encoder/layer_11/attention/self/key/kernel/bert/encoder/layer_11/attention/self/query/bias1bert/encoder/layer_11/attention/self/query/kernel/bert/encoder/layer_11/attention/self/value/bias1bert/encoder/layer_11/attention/self/value/kernel-bert/encoder/layer_11/intermediate/dense/bias/bert/encoder/layer_11/intermediate/dense/kernel+bert/encoder/layer_11/output/LayerNorm/beta,bert/encoder/layer_11/output/LayerNorm/gamma'bert/encoder/layer_11/output/dense/bias)bert/encoder/layer_11/output/dense/kernel4bert/encoder/layer_2/attention/output/LayerNorm/beta5bert/encoder/layer_2/attention/output/LayerNorm/gamma0bert/encoder/layer_2/attention/output/dense/bias2bert/encoder/layer_2/attention/output/dense/kernel,bert/encoder/layer_2/attention/self/key/bias.bert/encoder/layer_2/attention/self/key/kernel.bert/encoder/layer_2/attention/self/query/bias0bert/encoder/layer_2/attention/self/query/kernel.bert/encoder/layer_2/attention/self/value/bias0bert/encoder/layer_2/attention/self/value/kernel,bert/encoder/layer_2/intermediate/dense/bias.bert/encoder/layer_2/intermediate/dense/kernel*bert/encoder/layer_2/output/LayerNorm/beta+bert/encoder/layer_2/output/LayerNorm/gamma&bert/encoder/layer_2/output/dense/bias(bert/encoder/layer_2/output/dense/kernel4bert/encoder/layer_3/attention/output/LayerNorm/beta5bert/encoder/layer_3/attention/output/LayerNorm/gamma0bert/encoder/layer_3/attention/output/dense/bias2bert/encoder/layer_3/attention/output/dense/kernel,bert/encoder/layer_3/attention/self/key/bias.bert/encoder/layer_3/attention/self/key/kernel.bert/encoder/layer_3/attention/self/query/bias0bert/encoder/layer_3/attention/self/query/kernel.bert/encoder/layer_3/attention/self/value/bias0bert/encoder/layer_3/attention/self/value/kernel,bert/encoder/layer_3/intermediate/dense/bias.bert/encoder/layer_3/intermediate/dense/kernel*bert/encoder/layer_3/output/LayerNorm/beta+bert/encoder/layer_3/output/LayerNorm/gamma&bert/encoder/layer_3/output/dense/bias(bert/encoder/layer_3/output/dense/kernel4bert/encoder/layer_4/attention/output/LayerNorm/beta5bert/encoder/layer_4/attention/output/LayerNorm/gamma0bert/encoder/layer_4/attention/output/dense/bias2bert/encoder/layer_4/attention/output/dense/kernel,bert/encoder/layer_4/attention/self/key/bias.bert/encoder/layer_4/attention/self/key/kernel.bert/encoder/layer_4/attention/self/query/bias0bert/encoder/layer_4/attention/self/query/kernel.bert/encoder/layer_4/attention/self/value/bias0bert/encoder/layer_4/attention/self/value/kernel,bert/encoder/layer_4/intermediate/dense/bias.bert/encoder/layer_4/intermediate/dense/kernel*bert/encoder/layer_4/output/LayerNorm/beta+bert/encoder/layer_4/output/LayerNorm/gamma&bert/encoder/layer_4/output/dense/bias(bert/encoder/layer_4/output/dense/kernel4bert/encoder/layer_5/attention/output/LayerNorm/beta5bert/encoder/layer_5/attention/output/LayerNorm/gamma0bert/encoder/layer_5/attention/output/dense/bias2bert/encoder/layer_5/attention/output/dense/kernel,bert/encoder/layer_5/attention/self/key/bias.bert/encoder/layer_5/attention/self/key/kernel.bert/encoder/layer_5/attention/self/query/bias0bert/encoder/layer_5/attention/self/query/kernel.bert/encoder/layer_5/attention/self/value/bias0bert/encoder/layer_5/attention/self/value/kernel,bert/encoder/layer_5/intermediate/dense/bias.bert/encoder/layer_5/intermediate/dense/kernel*bert/encoder/layer_5/output/LayerNorm/beta+bert/encoder/layer_5/output/LayerNorm/gamma&bert/encoder/layer_5/output/dense/bias(bert/encoder/layer_5/output/dense/kernel4bert/encoder/layer_6/attention/output/LayerNorm/beta5bert/encoder/layer_6/attention/output/LayerNorm/gamma0bert/encoder/layer_6/attention/output/dense/bias2bert/encoder/layer_6/attention/output/dense/kernel,bert/encoder/layer_6/attention/self/key/bias.bert/encoder/layer_6/attention/self/key/kernel.bert/encoder/layer_6/attention/self/query/bias0bert/encoder/layer_6/attention/self/query/kernel.bert/encoder/layer_6/attention/self/value/bias0bert/encoder/layer_6/attention/self/value/kernel,bert/encoder/layer_6/intermediate/dense/bias.bert/encoder/layer_6/intermediate/dense/kernel*bert/encoder/layer_6/output/LayerNorm/beta+bert/encoder/layer_6/output/LayerNorm/gamma&bert/encoder/layer_6/output/dense/bias(bert/encoder/layer_6/output/dense/kernel4bert/encoder/layer_7/attention/output/LayerNorm/beta5bert/encoder/layer_7/attention/output/LayerNorm/gamma0bert/encoder/layer_7/attention/output/dense/bias2bert/encoder/layer_7/attention/output/dense/kernel,bert/encoder/layer_7/attention/self/key/bias.bert/encoder/layer_7/attention/self/key/kernel.bert/encoder/layer_7/attention/self/query/bias0bert/encoder/layer_7/attention/self/query/kernel.bert/encoder/layer_7/attention/self/value/bias0bert/encoder/layer_7/attention/self/value/kernel,bert/encoder/layer_7/intermediate/dense/bias.bert/encoder/layer_7/intermediate/dense/kernel*bert/encoder/layer_7/output/LayerNorm/beta+bert/encoder/layer_7/output/LayerNorm/gamma&bert/encoder/layer_7/output/dense/bias(bert/encoder/layer_7/output/dense/kernel4bert/encoder/layer_8/attention/output/LayerNorm/beta5bert/encoder/layer_8/attention/output/LayerNorm/gamma0bert/encoder/layer_8/attention/output/dense/bias2bert/encoder/layer_8/attention/output/dense/kernel,bert/encoder/layer_8/attention/self/key/bias.bert/encoder/layer_8/attention/self/key/kernel.bert/encoder/layer_8/attention/self/query/bias0bert/encoder/layer_8/attention/self/query/kernel.bert/encoder/layer_8/attention/self/value/bias0bert/encoder/layer_8/attention/self/value/kernel,bert/encoder/layer_8/intermediate/dense/bias.bert/encoder/layer_8/intermediate/dense/kernel*bert/encoder/layer_8/output/LayerNorm/beta+bert/encoder/layer_8/output/LayerNorm/gamma&bert/encoder/layer_8/output/dense/bias(bert/encoder/layer_8/output/dense/kernel4bert/encoder/layer_9/attention/output/LayerNorm/beta5bert/encoder/layer_9/attention/output/LayerNorm/gamma0bert/encoder/layer_9/attention/output/dense/bias2bert/encoder/layer_9/attention/output/dense/kernel,bert/encoder/layer_9/attention/self/key/bias.bert/encoder/layer_9/attention/self/key/kernel.bert/encoder/layer_9/attention/self/query/bias0bert/encoder/layer_9/attention/self/query/kernel.bert/encoder/layer_9/attention/self/value/bias0bert/encoder/layer_9/attention/self/value/kernel,bert/encoder/layer_9/intermediate/dense/bias.bert/encoder/layer_9/intermediate/dense/kernel*bert/encoder/layer_9/output/LayerNorm/beta+bert/encoder/layer_9/output/LayerNorm/gamma&bert/encoder/layer_9/output/dense/bias(bert/encoder/layer_9/output/dense/kernelbert/pooler/dense/biasbert/pooler/dense/kerneloutput_biasoutput_weights*Ú
dtypesÏ
Ì2É

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
_output_shapes
: *
T0*)
_class
loc:@save_1/ShardedFilename
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
îJ
save_1/RestoreV2/tensor_namesConst*J
valueJBJÉBbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelB5bert/encoder/layer_10/attention/output/LayerNorm/betaB6bert/encoder/layer_10/attention/output/LayerNorm/gammaB1bert/encoder/layer_10/attention/output/dense/biasB3bert/encoder/layer_10/attention/output/dense/kernelB-bert/encoder/layer_10/attention/self/key/biasB/bert/encoder/layer_10/attention/self/key/kernelB/bert/encoder/layer_10/attention/self/query/biasB1bert/encoder/layer_10/attention/self/query/kernelB/bert/encoder/layer_10/attention/self/value/biasB1bert/encoder/layer_10/attention/self/value/kernelB-bert/encoder/layer_10/intermediate/dense/biasB/bert/encoder/layer_10/intermediate/dense/kernelB+bert/encoder/layer_10/output/LayerNorm/betaB,bert/encoder/layer_10/output/LayerNorm/gammaB'bert/encoder/layer_10/output/dense/biasB)bert/encoder/layer_10/output/dense/kernelB5bert/encoder/layer_11/attention/output/LayerNorm/betaB6bert/encoder/layer_11/attention/output/LayerNorm/gammaB1bert/encoder/layer_11/attention/output/dense/biasB3bert/encoder/layer_11/attention/output/dense/kernelB-bert/encoder/layer_11/attention/self/key/biasB/bert/encoder/layer_11/attention/self/key/kernelB/bert/encoder/layer_11/attention/self/query/biasB1bert/encoder/layer_11/attention/self/query/kernelB/bert/encoder/layer_11/attention/self/value/biasB1bert/encoder/layer_11/attention/self/value/kernelB-bert/encoder/layer_11/intermediate/dense/biasB/bert/encoder/layer_11/intermediate/dense/kernelB+bert/encoder/layer_11/output/LayerNorm/betaB,bert/encoder/layer_11/output/LayerNorm/gammaB'bert/encoder/layer_11/output/dense/biasB)bert/encoder/layer_11/output/dense/kernelB4bert/encoder/layer_2/attention/output/LayerNorm/betaB5bert/encoder/layer_2/attention/output/LayerNorm/gammaB0bert/encoder/layer_2/attention/output/dense/biasB2bert/encoder/layer_2/attention/output/dense/kernelB,bert/encoder/layer_2/attention/self/key/biasB.bert/encoder/layer_2/attention/self/key/kernelB.bert/encoder/layer_2/attention/self/query/biasB0bert/encoder/layer_2/attention/self/query/kernelB.bert/encoder/layer_2/attention/self/value/biasB0bert/encoder/layer_2/attention/self/value/kernelB,bert/encoder/layer_2/intermediate/dense/biasB.bert/encoder/layer_2/intermediate/dense/kernelB*bert/encoder/layer_2/output/LayerNorm/betaB+bert/encoder/layer_2/output/LayerNorm/gammaB&bert/encoder/layer_2/output/dense/biasB(bert/encoder/layer_2/output/dense/kernelB4bert/encoder/layer_3/attention/output/LayerNorm/betaB5bert/encoder/layer_3/attention/output/LayerNorm/gammaB0bert/encoder/layer_3/attention/output/dense/biasB2bert/encoder/layer_3/attention/output/dense/kernelB,bert/encoder/layer_3/attention/self/key/biasB.bert/encoder/layer_3/attention/self/key/kernelB.bert/encoder/layer_3/attention/self/query/biasB0bert/encoder/layer_3/attention/self/query/kernelB.bert/encoder/layer_3/attention/self/value/biasB0bert/encoder/layer_3/attention/self/value/kernelB,bert/encoder/layer_3/intermediate/dense/biasB.bert/encoder/layer_3/intermediate/dense/kernelB*bert/encoder/layer_3/output/LayerNorm/betaB+bert/encoder/layer_3/output/LayerNorm/gammaB&bert/encoder/layer_3/output/dense/biasB(bert/encoder/layer_3/output/dense/kernelB4bert/encoder/layer_4/attention/output/LayerNorm/betaB5bert/encoder/layer_4/attention/output/LayerNorm/gammaB0bert/encoder/layer_4/attention/output/dense/biasB2bert/encoder/layer_4/attention/output/dense/kernelB,bert/encoder/layer_4/attention/self/key/biasB.bert/encoder/layer_4/attention/self/key/kernelB.bert/encoder/layer_4/attention/self/query/biasB0bert/encoder/layer_4/attention/self/query/kernelB.bert/encoder/layer_4/attention/self/value/biasB0bert/encoder/layer_4/attention/self/value/kernelB,bert/encoder/layer_4/intermediate/dense/biasB.bert/encoder/layer_4/intermediate/dense/kernelB*bert/encoder/layer_4/output/LayerNorm/betaB+bert/encoder/layer_4/output/LayerNorm/gammaB&bert/encoder/layer_4/output/dense/biasB(bert/encoder/layer_4/output/dense/kernelB4bert/encoder/layer_5/attention/output/LayerNorm/betaB5bert/encoder/layer_5/attention/output/LayerNorm/gammaB0bert/encoder/layer_5/attention/output/dense/biasB2bert/encoder/layer_5/attention/output/dense/kernelB,bert/encoder/layer_5/attention/self/key/biasB.bert/encoder/layer_5/attention/self/key/kernelB.bert/encoder/layer_5/attention/self/query/biasB0bert/encoder/layer_5/attention/self/query/kernelB.bert/encoder/layer_5/attention/self/value/biasB0bert/encoder/layer_5/attention/self/value/kernelB,bert/encoder/layer_5/intermediate/dense/biasB.bert/encoder/layer_5/intermediate/dense/kernelB*bert/encoder/layer_5/output/LayerNorm/betaB+bert/encoder/layer_5/output/LayerNorm/gammaB&bert/encoder/layer_5/output/dense/biasB(bert/encoder/layer_5/output/dense/kernelB4bert/encoder/layer_6/attention/output/LayerNorm/betaB5bert/encoder/layer_6/attention/output/LayerNorm/gammaB0bert/encoder/layer_6/attention/output/dense/biasB2bert/encoder/layer_6/attention/output/dense/kernelB,bert/encoder/layer_6/attention/self/key/biasB.bert/encoder/layer_6/attention/self/key/kernelB.bert/encoder/layer_6/attention/self/query/biasB0bert/encoder/layer_6/attention/self/query/kernelB.bert/encoder/layer_6/attention/self/value/biasB0bert/encoder/layer_6/attention/self/value/kernelB,bert/encoder/layer_6/intermediate/dense/biasB.bert/encoder/layer_6/intermediate/dense/kernelB*bert/encoder/layer_6/output/LayerNorm/betaB+bert/encoder/layer_6/output/LayerNorm/gammaB&bert/encoder/layer_6/output/dense/biasB(bert/encoder/layer_6/output/dense/kernelB4bert/encoder/layer_7/attention/output/LayerNorm/betaB5bert/encoder/layer_7/attention/output/LayerNorm/gammaB0bert/encoder/layer_7/attention/output/dense/biasB2bert/encoder/layer_7/attention/output/dense/kernelB,bert/encoder/layer_7/attention/self/key/biasB.bert/encoder/layer_7/attention/self/key/kernelB.bert/encoder/layer_7/attention/self/query/biasB0bert/encoder/layer_7/attention/self/query/kernelB.bert/encoder/layer_7/attention/self/value/biasB0bert/encoder/layer_7/attention/self/value/kernelB,bert/encoder/layer_7/intermediate/dense/biasB.bert/encoder/layer_7/intermediate/dense/kernelB*bert/encoder/layer_7/output/LayerNorm/betaB+bert/encoder/layer_7/output/LayerNorm/gammaB&bert/encoder/layer_7/output/dense/biasB(bert/encoder/layer_7/output/dense/kernelB4bert/encoder/layer_8/attention/output/LayerNorm/betaB5bert/encoder/layer_8/attention/output/LayerNorm/gammaB0bert/encoder/layer_8/attention/output/dense/biasB2bert/encoder/layer_8/attention/output/dense/kernelB,bert/encoder/layer_8/attention/self/key/biasB.bert/encoder/layer_8/attention/self/key/kernelB.bert/encoder/layer_8/attention/self/query/biasB0bert/encoder/layer_8/attention/self/query/kernelB.bert/encoder/layer_8/attention/self/value/biasB0bert/encoder/layer_8/attention/self/value/kernelB,bert/encoder/layer_8/intermediate/dense/biasB.bert/encoder/layer_8/intermediate/dense/kernelB*bert/encoder/layer_8/output/LayerNorm/betaB+bert/encoder/layer_8/output/LayerNorm/gammaB&bert/encoder/layer_8/output/dense/biasB(bert/encoder/layer_8/output/dense/kernelB4bert/encoder/layer_9/attention/output/LayerNorm/betaB5bert/encoder/layer_9/attention/output/LayerNorm/gammaB0bert/encoder/layer_9/attention/output/dense/biasB2bert/encoder/layer_9/attention/output/dense/kernelB,bert/encoder/layer_9/attention/self/key/biasB.bert/encoder/layer_9/attention/self/key/kernelB.bert/encoder/layer_9/attention/self/query/biasB0bert/encoder/layer_9/attention/self/query/kernelB.bert/encoder/layer_9/attention/self/value/biasB0bert/encoder/layer_9/attention/self/value/kernelB,bert/encoder/layer_9/intermediate/dense/biasB.bert/encoder/layer_9/intermediate/dense/kernelB*bert/encoder/layer_9/output/LayerNorm/betaB+bert/encoder/layer_9/output/LayerNorm/gammaB&bert/encoder/layer_9/output/dense/biasB(bert/encoder/layer_9/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelBoutput_biasBoutput_weights*
dtype0*
_output_shapes	
:É
ÿ
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes	
:É*¨
valueBÉB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*º
_output_shapes§
¤:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Ú
dtypesÏ
Ì2É
Ë
save_1/AssignAssignbert/embeddings/LayerNorm/betasave_1/RestoreV2*
use_locking(*
T0*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
Ñ
save_1/Assign_1Assignbert/embeddings/LayerNorm/gammasave_1/RestoreV2:1*
T0*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
Þ
save_1/Assign_2Assign#bert/embeddings/position_embeddingssave_1/RestoreV2:2*6
_class,
*(loc:@bert/embeddings/position_embeddings*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
á
save_1/Assign_3Assign%bert/embeddings/token_type_embeddingssave_1/RestoreV2:3*
use_locking(*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
validate_shape(*
_output_shapes
:	
×
save_1/Assign_4Assignbert/embeddings/word_embeddingssave_1/RestoreV2:4*
use_locking(*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*
validate_shape(*!
_output_shapes
:ºî
û
save_1/Assign_5Assign4bert/encoder/layer_0/attention/output/LayerNorm/betasave_1/RestoreV2:5*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ý
save_1/Assign_6Assign5bert/encoder/layer_0/attention/output/LayerNorm/gammasave_1/RestoreV2:6*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ó
save_1/Assign_7Assign0bert/encoder/layer_0/attention/output/dense/biassave_1/RestoreV2:7*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ü
save_1/Assign_8Assign2bert/encoder/layer_0/attention/output/dense/kernelsave_1/RestoreV2:8*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ë
save_1/Assign_9Assign,bert/encoder/layer_0/attention/self/key/biassave_1/RestoreV2:9*
T0*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
save_1/Assign_10Assign.bert/encoder/layer_0/attention/self/key/kernelsave_1/RestoreV2:10*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ñ
save_1/Assign_11Assign.bert/encoder/layer_0/attention/self/query/biassave_1/RestoreV2:11*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_12Assign0bert/encoder/layer_0/attention/self/query/kernelsave_1/RestoreV2:12*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ñ
save_1/Assign_13Assign.bert/encoder/layer_0/attention/self/value/biassave_1/RestoreV2:13*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
validate_shape(
ú
save_1/Assign_14Assign0bert/encoder/layer_0/attention/self/value/kernelsave_1/RestoreV2:14*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel
í
save_1/Assign_15Assign,bert/encoder/layer_0/intermediate/dense/biassave_1/RestoreV2:15*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_16Assign.bert/encoder/layer_0/intermediate/dense/kernelsave_1/RestoreV2:16*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
é
save_1/Assign_17Assign*bert/encoder/layer_0/output/LayerNorm/betasave_1/RestoreV2:17*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
validate_shape(
ë
save_1/Assign_18Assign+bert/encoder/layer_0/output/LayerNorm/gammasave_1/RestoreV2:18*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
validate_shape(
á
save_1/Assign_19Assign&bert/encoder/layer_0/output/dense/biassave_1/RestoreV2:19*
_output_shapes	
:*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
validate_shape(
ê
save_1/Assign_20Assign(bert/encoder/layer_0/output/dense/kernelsave_1/RestoreV2:20*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ý
save_1/Assign_21Assign4bert/encoder/layer_1/attention/output/LayerNorm/betasave_1/RestoreV2:21*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ÿ
save_1/Assign_22Assign5bert/encoder/layer_1/attention/output/LayerNorm/gammasave_1/RestoreV2:22*
T0*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
õ
save_1/Assign_23Assign0bert/encoder/layer_1/attention/output/dense/biassave_1/RestoreV2:23*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
þ
save_1/Assign_24Assign2bert/encoder/layer_1/attention/output/dense/kernelsave_1/RestoreV2:24*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

í
save_1/Assign_25Assign,bert/encoder/layer_1/attention/self/key/biassave_1/RestoreV2:25*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_26Assign.bert/encoder/layer_1/attention/self/key/kernelsave_1/RestoreV2:26*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ñ
save_1/Assign_27Assign.bert/encoder/layer_1/attention/self/query/biassave_1/RestoreV2:27*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_28Assign0bert/encoder/layer_1/attention/self/query/kernelsave_1/RestoreV2:28*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ñ
save_1/Assign_29Assign.bert/encoder/layer_1/attention/self/value/biassave_1/RestoreV2:29*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ú
save_1/Assign_30Assign0bert/encoder/layer_1/attention/self/value/kernelsave_1/RestoreV2:30*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

í
save_1/Assign_31Assign,bert/encoder/layer_1/intermediate/dense/biassave_1/RestoreV2:31*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias
ö
save_1/Assign_32Assign.bert/encoder/layer_1/intermediate/dense/kernelsave_1/RestoreV2:32*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel
é
save_1/Assign_33Assign*bert/encoder/layer_1/output/LayerNorm/betasave_1/RestoreV2:33*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ë
save_1/Assign_34Assign+bert/encoder/layer_1/output/LayerNorm/gammasave_1/RestoreV2:34*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
á
save_1/Assign_35Assign&bert/encoder/layer_1/output/dense/biassave_1/RestoreV2:35*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
validate_shape(*
_output_shapes	
:
ê
save_1/Assign_36Assign(bert/encoder/layer_1/output/dense/kernelsave_1/RestoreV2:36*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel
ÿ
save_1/Assign_37Assign5bert/encoder/layer_10/attention/output/LayerNorm/betasave_1/RestoreV2:37*
T0*H
_class>
<:loc:@bert/encoder/layer_10/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_38Assign6bert/encoder/layer_10/attention/output/LayerNorm/gammasave_1/RestoreV2:38*
_output_shapes	
:*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_10/attention/output/LayerNorm/gamma*
validate_shape(
÷
save_1/Assign_39Assign1bert/encoder/layer_10/attention/output/dense/biassave_1/RestoreV2:39*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/output/dense/bias

save_1/Assign_40Assign3bert/encoder/layer_10/attention/output/dense/kernelsave_1/RestoreV2:40*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_10/attention/output/dense/kernel
ï
save_1/Assign_41Assign-bert/encoder/layer_10/attention/self/key/biassave_1/RestoreV2:41*@
_class6
42loc:@bert/encoder/layer_10/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save_1/Assign_42Assign/bert/encoder/layer_10/attention/self/key/kernelsave_1/RestoreV2:42*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_43Assign/bert/encoder/layer_10/attention/self/query/biassave_1/RestoreV2:43*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ü
save_1/Assign_44Assign1bert/encoder/layer_10/attention/self/query/kernelsave_1/RestoreV2:44*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_45Assign/bert/encoder/layer_10/attention/self/value/biassave_1/RestoreV2:45*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_10/attention/self/value/bias*
validate_shape(
ü
save_1/Assign_46Assign1bert/encoder/layer_10/attention/self/value/kernelsave_1/RestoreV2:46*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_10/attention/self/value/kernel
ï
save_1/Assign_47Assign-bert/encoder/layer_10/intermediate/dense/biassave_1/RestoreV2:47*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_10/intermediate/dense/bias*
validate_shape(
ø
save_1/Assign_48Assign/bert/encoder/layer_10/intermediate/dense/kernelsave_1/RestoreV2:48*
T0*B
_class8
64loc:@bert/encoder/layer_10/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ë
save_1/Assign_49Assign+bert/encoder/layer_10/output/LayerNorm/betasave_1/RestoreV2:49*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_10/output/LayerNorm/beta
í
save_1/Assign_50Assign,bert/encoder/layer_10/output/LayerNorm/gammasave_1/RestoreV2:50*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_10/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ã
save_1/Assign_51Assign'bert/encoder/layer_10/output/dense/biassave_1/RestoreV2:51*
use_locking(*
T0*:
_class0
.,loc:@bert/encoder/layer_10/output/dense/bias*
validate_shape(*
_output_shapes	
:
ì
save_1/Assign_52Assign)bert/encoder/layer_10/output/dense/kernelsave_1/RestoreV2:52*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*<
_class2
0.loc:@bert/encoder/layer_10/output/dense/kernel
ÿ
save_1/Assign_53Assign5bert/encoder/layer_11/attention/output/LayerNorm/betasave_1/RestoreV2:53*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_11/attention/output/LayerNorm/beta

save_1/Assign_54Assign6bert/encoder/layer_11/attention/output/LayerNorm/gammasave_1/RestoreV2:54*
use_locking(*
T0*I
_class?
=;loc:@bert/encoder/layer_11/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
÷
save_1/Assign_55Assign1bert/encoder/layer_11/attention/output/dense/biassave_1/RestoreV2:55*D
_class:
86loc:@bert/encoder/layer_11/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_56Assign3bert/encoder/layer_11/attention/output/dense/kernelsave_1/RestoreV2:56*
use_locking(*
T0*F
_class<
:8loc:@bert/encoder/layer_11/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ï
save_1/Assign_57Assign-bert/encoder/layer_11/attention/self/key/biassave_1/RestoreV2:57*
_output_shapes	
:*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/attention/self/key/bias*
validate_shape(
ø
save_1/Assign_58Assign/bert/encoder/layer_11/attention/self/key/kernelsave_1/RestoreV2:58*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_59Assign/bert/encoder/layer_11/attention/self/query/biassave_1/RestoreV2:59*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ü
save_1/Assign_60Assign1bert/encoder/layer_11/attention/self/query/kernelsave_1/RestoreV2:60*
use_locking(*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_61Assign/bert/encoder/layer_11/attention/self/value/biassave_1/RestoreV2:61*
_output_shapes	
:*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/attention/self/value/bias*
validate_shape(
ü
save_1/Assign_62Assign1bert/encoder/layer_11/attention/self/value/kernelsave_1/RestoreV2:62*
T0*D
_class:
86loc:@bert/encoder/layer_11/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ï
save_1/Assign_63Assign-bert/encoder/layer_11/intermediate/dense/biassave_1/RestoreV2:63*
use_locking(*
T0*@
_class6
42loc:@bert/encoder/layer_11/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_64Assign/bert/encoder/layer_11/intermediate/dense/kernelsave_1/RestoreV2:64*
use_locking(*
T0*B
_class8
64loc:@bert/encoder/layer_11/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

ë
save_1/Assign_65Assign+bert/encoder/layer_11/output/LayerNorm/betasave_1/RestoreV2:65*>
_class4
20loc:@bert/encoder/layer_11/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
í
save_1/Assign_66Assign,bert/encoder/layer_11/output/LayerNorm/gammasave_1/RestoreV2:66*
T0*?
_class5
31loc:@bert/encoder/layer_11/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
ã
save_1/Assign_67Assign'bert/encoder/layer_11/output/dense/biassave_1/RestoreV2:67*
T0*:
_class0
.,loc:@bert/encoder/layer_11/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ì
save_1/Assign_68Assign)bert/encoder/layer_11/output/dense/kernelsave_1/RestoreV2:68*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*<
_class2
0.loc:@bert/encoder/layer_11/output/dense/kernel
ý
save_1/Assign_69Assign4bert/encoder/layer_2/attention/output/LayerNorm/betasave_1/RestoreV2:69*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_2/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ÿ
save_1/Assign_70Assign5bert/encoder/layer_2/attention/output/LayerNorm/gammasave_1/RestoreV2:70*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_2/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
õ
save_1/Assign_71Assign0bert/encoder/layer_2/attention/output/dense/biassave_1/RestoreV2:71*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/output/dense/bias*
validate_shape(
þ
save_1/Assign_72Assign2bert/encoder/layer_2/attention/output/dense/kernelsave_1/RestoreV2:72*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_2/attention/output/dense/kernel
í
save_1/Assign_73Assign,bert/encoder/layer_2/attention/self/key/biassave_1/RestoreV2:73*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_74Assign.bert/encoder/layer_2/attention/self/key/kernelsave_1/RestoreV2:74*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ñ
save_1/Assign_75Assign.bert/encoder/layer_2/attention/self/query/biassave_1/RestoreV2:75*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_2/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_76Assign0bert/encoder/layer_2/attention/self/query/kernelsave_1/RestoreV2:76*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ñ
save_1/Assign_77Assign.bert/encoder/layer_2/attention/self/value/biassave_1/RestoreV2:77*A
_class7
53loc:@bert/encoder/layer_2/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ú
save_1/Assign_78Assign0bert/encoder/layer_2/attention/self/value/kernelsave_1/RestoreV2:78*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_2/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

í
save_1/Assign_79Assign,bert/encoder/layer_2/intermediate/dense/biassave_1/RestoreV2:79*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_2/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_80Assign.bert/encoder/layer_2/intermediate/dense/kernelsave_1/RestoreV2:80*A
_class7
53loc:@bert/encoder/layer_2/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
é
save_1/Assign_81Assign*bert/encoder/layer_2/output/LayerNorm/betasave_1/RestoreV2:81*
T0*=
_class3
1/loc:@bert/encoder/layer_2/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
ë
save_1/Assign_82Assign+bert/encoder/layer_2/output/LayerNorm/gammasave_1/RestoreV2:82*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_2/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
á
save_1/Assign_83Assign&bert/encoder/layer_2/output/dense/biassave_1/RestoreV2:83*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_2/output/dense/bias*
validate_shape(*
_output_shapes	
:
ê
save_1/Assign_84Assign(bert/encoder/layer_2/output/dense/kernelsave_1/RestoreV2:84*;
_class1
/-loc:@bert/encoder/layer_2/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ý
save_1/Assign_85Assign4bert/encoder/layer_3/attention/output/LayerNorm/betasave_1/RestoreV2:85*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_3/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
ÿ
save_1/Assign_86Assign5bert/encoder/layer_3/attention/output/LayerNorm/gammasave_1/RestoreV2:86*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_3/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
õ
save_1/Assign_87Assign0bert/encoder/layer_3/attention/output/dense/biassave_1/RestoreV2:87*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:
þ
save_1/Assign_88Assign2bert/encoder/layer_3/attention/output/dense/kernelsave_1/RestoreV2:88*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_3/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

í
save_1/Assign_89Assign,bert/encoder/layer_3/attention/self/key/biassave_1/RestoreV2:89*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_3/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ö
save_1/Assign_90Assign.bert/encoder/layer_3/attention/self/key/kernelsave_1/RestoreV2:90*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ñ
save_1/Assign_91Assign.bert/encoder/layer_3/attention/self/query/biassave_1/RestoreV2:91*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_92Assign0bert/encoder/layer_3/attention/self/query/kernelsave_1/RestoreV2:92*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ñ
save_1/Assign_93Assign.bert/encoder/layer_3/attention/self/value/biassave_1/RestoreV2:93*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
ú
save_1/Assign_94Assign0bert/encoder/layer_3/attention/self/value/kernelsave_1/RestoreV2:94*
T0*C
_class9
75loc:@bert/encoder/layer_3/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
í
save_1/Assign_95Assign,bert/encoder/layer_3/intermediate/dense/biassave_1/RestoreV2:95*
T0*?
_class5
31loc:@bert/encoder/layer_3/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ö
save_1/Assign_96Assign.bert/encoder/layer_3/intermediate/dense/kernelsave_1/RestoreV2:96*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_3/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

é
save_1/Assign_97Assign*bert/encoder/layer_3/output/LayerNorm/betasave_1/RestoreV2:97*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_3/output/LayerNorm/beta*
validate_shape(
ë
save_1/Assign_98Assign+bert/encoder/layer_3/output/LayerNorm/gammasave_1/RestoreV2:98*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_3/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
á
save_1/Assign_99Assign&bert/encoder/layer_3/output/dense/biassave_1/RestoreV2:99*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_3/output/dense/bias*
validate_shape(*
_output_shapes	
:
ì
save_1/Assign_100Assign(bert/encoder/layer_3/output/dense/kernelsave_1/RestoreV2:100*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_3/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ÿ
save_1/Assign_101Assign4bert/encoder/layer_4/attention/output/LayerNorm/betasave_1/RestoreV2:101*
_output_shapes	
:*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_4/attention/output/LayerNorm/beta*
validate_shape(

save_1/Assign_102Assign5bert/encoder/layer_4/attention/output/LayerNorm/gammasave_1/RestoreV2:102*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_4/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
÷
save_1/Assign_103Assign0bert/encoder/layer_4/attention/output/dense/biassave_1/RestoreV2:103*C
_class9
75loc:@bert/encoder/layer_4/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_104Assign2bert/encoder/layer_4/attention/output/dense/kernelsave_1/RestoreV2:104*E
_class;
97loc:@bert/encoder/layer_4/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ï
save_1/Assign_105Assign,bert/encoder/layer_4/attention/self/key/biassave_1/RestoreV2:105*
T0*?
_class5
31loc:@bert/encoder/layer_4/attention/self/key/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ø
save_1/Assign_106Assign.bert/encoder/layer_4/attention/self/key/kernelsave_1/RestoreV2:106*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_107Assign.bert/encoder/layer_4/attention/self/query/biassave_1/RestoreV2:107*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/query/bias
ü
save_1/Assign_108Assign0bert/encoder/layer_4/attention/self/query/kernelsave_1/RestoreV2:108*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_109Assign.bert/encoder/layer_4/attention/self/value/biassave_1/RestoreV2:109*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
ü
save_1/Assign_110Assign0bert/encoder/layer_4/attention/self/value/kernelsave_1/RestoreV2:110*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_4/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:

ï
save_1/Assign_111Assign,bert/encoder/layer_4/intermediate/dense/biassave_1/RestoreV2:111*?
_class5
31loc:@bert/encoder/layer_4/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save_1/Assign_112Assign.bert/encoder/layer_4/intermediate/dense/kernelsave_1/RestoreV2:112*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_4/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

ë
save_1/Assign_113Assign*bert/encoder/layer_4/output/LayerNorm/betasave_1/RestoreV2:113*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_4/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
í
save_1/Assign_114Assign+bert/encoder/layer_4/output/LayerNorm/gammasave_1/RestoreV2:114*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_4/output/LayerNorm/gamma
ã
save_1/Assign_115Assign&bert/encoder/layer_4/output/dense/biassave_1/RestoreV2:115*
_output_shapes	
:*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_4/output/dense/bias*
validate_shape(
ì
save_1/Assign_116Assign(bert/encoder/layer_4/output/dense/kernelsave_1/RestoreV2:116*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_4/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ÿ
save_1/Assign_117Assign4bert/encoder/layer_5/attention/output/LayerNorm/betasave_1/RestoreV2:117*
T0*G
_class=
;9loc:@bert/encoder/layer_5/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_118Assign5bert/encoder/layer_5/attention/output/LayerNorm/gammasave_1/RestoreV2:118*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_5/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
÷
save_1/Assign_119Assign0bert/encoder/layer_5/attention/output/dense/biassave_1/RestoreV2:119*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/output/dense/bias

save_1/Assign_120Assign2bert/encoder/layer_5/attention/output/dense/kernelsave_1/RestoreV2:120* 
_output_shapes
:
*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_5/attention/output/dense/kernel*
validate_shape(
ï
save_1/Assign_121Assign,bert/encoder/layer_5/attention/self/key/biassave_1/RestoreV2:121*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_5/attention/self/key/bias
ø
save_1/Assign_122Assign.bert/encoder/layer_5/attention/self/key/kernelsave_1/RestoreV2:122*A
_class7
53loc:@bert/encoder/layer_5/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ó
save_1/Assign_123Assign.bert/encoder/layer_5/attention/self/query/biassave_1/RestoreV2:123*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ü
save_1/Assign_124Assign0bert/encoder/layer_5/attention/self/query/kernelsave_1/RestoreV2:124*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_125Assign.bert/encoder/layer_5/attention/self/value/biassave_1/RestoreV2:125*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
ü
save_1/Assign_126Assign0bert/encoder/layer_5/attention/self/value/kernelsave_1/RestoreV2:126* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_5/attention/self/value/kernel*
validate_shape(
ï
save_1/Assign_127Assign,bert/encoder/layer_5/intermediate/dense/biassave_1/RestoreV2:127*?
_class5
31loc:@bert/encoder/layer_5/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save_1/Assign_128Assign.bert/encoder/layer_5/intermediate/dense/kernelsave_1/RestoreV2:128*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_5/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

ë
save_1/Assign_129Assign*bert/encoder/layer_5/output/LayerNorm/betasave_1/RestoreV2:129*
_output_shapes	
:*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_5/output/LayerNorm/beta*
validate_shape(
í
save_1/Assign_130Assign+bert/encoder/layer_5/output/LayerNorm/gammasave_1/RestoreV2:130*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_5/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ã
save_1/Assign_131Assign&bert/encoder/layer_5/output/dense/biassave_1/RestoreV2:131*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_5/output/dense/bias*
validate_shape(*
_output_shapes	
:
ì
save_1/Assign_132Assign(bert/encoder/layer_5/output/dense/kernelsave_1/RestoreV2:132*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_5/output/dense/kernel
ÿ
save_1/Assign_133Assign4bert/encoder/layer_6/attention/output/LayerNorm/betasave_1/RestoreV2:133*G
_class=
;9loc:@bert/encoder/layer_6/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_134Assign5bert/encoder/layer_6/attention/output/LayerNorm/gammasave_1/RestoreV2:134*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_6/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
÷
save_1/Assign_135Assign0bert/encoder/layer_6/attention/output/dense/biassave_1/RestoreV2:135*C
_class9
75loc:@bert/encoder/layer_6/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

save_1/Assign_136Assign2bert/encoder/layer_6/attention/output/dense/kernelsave_1/RestoreV2:136*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_6/attention/output/dense/kernel
ï
save_1/Assign_137Assign,bert/encoder/layer_6/attention/self/key/biassave_1/RestoreV2:137*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_6/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_138Assign.bert/encoder/layer_6/attention/self/key/kernelsave_1/RestoreV2:138* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/key/kernel*
validate_shape(
ó
save_1/Assign_139Assign.bert/encoder/layer_6/attention/self/query/biassave_1/RestoreV2:139*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/query/bias*
validate_shape(*
_output_shapes	
:
ü
save_1/Assign_140Assign0bert/encoder/layer_6/attention/self/query/kernelsave_1/RestoreV2:140*C
_class9
75loc:@bert/encoder/layer_6/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ó
save_1/Assign_141Assign.bert/encoder/layer_6/attention/self/value/biassave_1/RestoreV2:141*
T0*A
_class7
53loc:@bert/encoder/layer_6/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ü
save_1/Assign_142Assign0bert/encoder/layer_6/attention/self/value/kernelsave_1/RestoreV2:142*
T0*C
_class9
75loc:@bert/encoder/layer_6/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ï
save_1/Assign_143Assign,bert/encoder/layer_6/intermediate/dense/biassave_1/RestoreV2:143*?
_class5
31loc:@bert/encoder/layer_6/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ø
save_1/Assign_144Assign.bert/encoder/layer_6/intermediate/dense/kernelsave_1/RestoreV2:144*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_6/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:

ë
save_1/Assign_145Assign*bert/encoder/layer_6/output/LayerNorm/betasave_1/RestoreV2:145*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_6/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
í
save_1/Assign_146Assign+bert/encoder/layer_6/output/LayerNorm/gammasave_1/RestoreV2:146*>
_class4
20loc:@bert/encoder/layer_6/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ã
save_1/Assign_147Assign&bert/encoder/layer_6/output/dense/biassave_1/RestoreV2:147*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*9
_class/
-+loc:@bert/encoder/layer_6/output/dense/bias
ì
save_1/Assign_148Assign(bert/encoder/layer_6/output/dense/kernelsave_1/RestoreV2:148*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_6/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ÿ
save_1/Assign_149Assign4bert/encoder/layer_7/attention/output/LayerNorm/betasave_1/RestoreV2:149*
_output_shapes	
:*
use_locking(*
T0*G
_class=
;9loc:@bert/encoder/layer_7/attention/output/LayerNorm/beta*
validate_shape(

save_1/Assign_150Assign5bert/encoder/layer_7/attention/output/LayerNorm/gammasave_1/RestoreV2:150*
_output_shapes	
:*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_7/attention/output/LayerNorm/gamma*
validate_shape(
÷
save_1/Assign_151Assign0bert/encoder/layer_7/attention/output/dense/biassave_1/RestoreV2:151*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:

save_1/Assign_152Assign2bert/encoder/layer_7/attention/output/dense/kernelsave_1/RestoreV2:152*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_7/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ï
save_1/Assign_153Assign,bert/encoder/layer_7/attention/self/key/biassave_1/RestoreV2:153*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_154Assign.bert/encoder/layer_7/attention/self/key/kernelsave_1/RestoreV2:154*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_155Assign.bert/encoder/layer_7/attention/self/query/biassave_1/RestoreV2:155*A
_class7
53loc:@bert/encoder/layer_7/attention/self/query/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ü
save_1/Assign_156Assign0bert/encoder/layer_7/attention/self/query/kernelsave_1/RestoreV2:156*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_157Assign.bert/encoder/layer_7/attention/self/value/biassave_1/RestoreV2:157*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/attention/self/value/bias*
validate_shape(
ü
save_1/Assign_158Assign0bert/encoder/layer_7/attention/self/value/kernelsave_1/RestoreV2:158*
T0*C
_class9
75loc:@bert/encoder/layer_7/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ï
save_1/Assign_159Assign,bert/encoder/layer_7/intermediate/dense/biassave_1/RestoreV2:159*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_7/intermediate/dense/bias*
validate_shape(
ø
save_1/Assign_160Assign.bert/encoder/layer_7/intermediate/dense/kernelsave_1/RestoreV2:160*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_7/intermediate/dense/kernel
ë
save_1/Assign_161Assign*bert/encoder/layer_7/output/LayerNorm/betasave_1/RestoreV2:161*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_7/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
í
save_1/Assign_162Assign+bert/encoder/layer_7/output/LayerNorm/gammasave_1/RestoreV2:162*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_7/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ã
save_1/Assign_163Assign&bert/encoder/layer_7/output/dense/biassave_1/RestoreV2:163*
T0*9
_class/
-+loc:@bert/encoder/layer_7/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ì
save_1/Assign_164Assign(bert/encoder/layer_7/output/dense/kernelsave_1/RestoreV2:164*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_7/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ÿ
save_1/Assign_165Assign4bert/encoder/layer_8/attention/output/LayerNorm/betasave_1/RestoreV2:165*
T0*G
_class=
;9loc:@bert/encoder/layer_8/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_166Assign5bert/encoder/layer_8/attention/output/LayerNorm/gammasave_1/RestoreV2:166*
use_locking(*
T0*H
_class>
<:loc:@bert/encoder/layer_8/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
÷
save_1/Assign_167Assign0bert/encoder/layer_8/attention/output/dense/biassave_1/RestoreV2:167*
_output_shapes	
:*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/output/dense/bias*
validate_shape(

save_1/Assign_168Assign2bert/encoder/layer_8/attention/output/dense/kernelsave_1/RestoreV2:168*
T0*E
_class;
97loc:@bert/encoder/layer_8/attention/output/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ï
save_1/Assign_169Assign,bert/encoder/layer_8/attention/self/key/biassave_1/RestoreV2:169*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_8/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_170Assign.bert/encoder/layer_8/attention/self/key/kernelsave_1/RestoreV2:170* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/key/kernel*
validate_shape(
ó
save_1/Assign_171Assign.bert/encoder/layer_8/attention/self/query/biassave_1/RestoreV2:171*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/query/bias*
validate_shape(
ü
save_1/Assign_172Assign0bert/encoder/layer_8/attention/self/query/kernelsave_1/RestoreV2:172*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_8/attention/self/query/kernel*
validate_shape(* 
_output_shapes
:

ó
save_1/Assign_173Assign.bert/encoder/layer_8/attention/self/value/biassave_1/RestoreV2:173*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_8/attention/self/value/bias*
validate_shape(*
_output_shapes	
:
ü
save_1/Assign_174Assign0bert/encoder/layer_8/attention/self/value/kernelsave_1/RestoreV2:174*C
_class9
75loc:@bert/encoder/layer_8/attention/self/value/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
ï
save_1/Assign_175Assign,bert/encoder/layer_8/intermediate/dense/biassave_1/RestoreV2:175*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_8/intermediate/dense/bias
ø
save_1/Assign_176Assign.bert/encoder/layer_8/intermediate/dense/kernelsave_1/RestoreV2:176*
T0*A
_class7
53loc:@bert/encoder/layer_8/intermediate/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ë
save_1/Assign_177Assign*bert/encoder/layer_8/output/LayerNorm/betasave_1/RestoreV2:177*
use_locking(*
T0*=
_class3
1/loc:@bert/encoder/layer_8/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:
í
save_1/Assign_178Assign+bert/encoder/layer_8/output/LayerNorm/gammasave_1/RestoreV2:178*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_8/output/LayerNorm/gamma
ã
save_1/Assign_179Assign&bert/encoder/layer_8/output/dense/biassave_1/RestoreV2:179*
T0*9
_class/
-+loc:@bert/encoder/layer_8/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ì
save_1/Assign_180Assign(bert/encoder/layer_8/output/dense/kernelsave_1/RestoreV2:180*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_8/output/dense/kernel*
validate_shape(* 
_output_shapes
:

ÿ
save_1/Assign_181Assign4bert/encoder/layer_9/attention/output/LayerNorm/betasave_1/RestoreV2:181*
T0*G
_class=
;9loc:@bert/encoder/layer_9/attention/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(

save_1/Assign_182Assign5bert/encoder/layer_9/attention/output/LayerNorm/gammasave_1/RestoreV2:182*
T0*H
_class>
<:loc:@bert/encoder/layer_9/attention/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:*
use_locking(
÷
save_1/Assign_183Assign0bert/encoder/layer_9/attention/output/dense/biassave_1/RestoreV2:183*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/output/dense/bias*
validate_shape(*
_output_shapes	
:

save_1/Assign_184Assign2bert/encoder/layer_9/attention/output/dense/kernelsave_1/RestoreV2:184* 
_output_shapes
:
*
use_locking(*
T0*E
_class;
97loc:@bert/encoder/layer_9/attention/output/dense/kernel*
validate_shape(
ï
save_1/Assign_185Assign,bert/encoder/layer_9/attention/self/key/biassave_1/RestoreV2:185*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/attention/self/key/bias*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_186Assign.bert/encoder/layer_9/attention/self/key/kernelsave_1/RestoreV2:186*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/key/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
ó
save_1/Assign_187Assign.bert/encoder/layer_9/attention/self/query/biassave_1/RestoreV2:187*
_output_shapes	
:*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/attention/self/query/bias*
validate_shape(
ü
save_1/Assign_188Assign0bert/encoder/layer_9/attention/self/query/kernelsave_1/RestoreV2:188*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/query/kernel
ó
save_1/Assign_189Assign.bert/encoder/layer_9/attention/self/value/biassave_1/RestoreV2:189*A
_class7
53loc:@bert/encoder/layer_9/attention/self/value/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
ü
save_1/Assign_190Assign0bert/encoder/layer_9/attention/self/value/kernelsave_1/RestoreV2:190*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*C
_class9
75loc:@bert/encoder/layer_9/attention/self/value/kernel
ï
save_1/Assign_191Assign,bert/encoder/layer_9/intermediate/dense/biassave_1/RestoreV2:191*
use_locking(*
T0*?
_class5
31loc:@bert/encoder/layer_9/intermediate/dense/bias*
validate_shape(*
_output_shapes	
:
ø
save_1/Assign_192Assign.bert/encoder/layer_9/intermediate/dense/kernelsave_1/RestoreV2:192* 
_output_shapes
:
*
use_locking(*
T0*A
_class7
53loc:@bert/encoder/layer_9/intermediate/dense/kernel*
validate_shape(
ë
save_1/Assign_193Assign*bert/encoder/layer_9/output/LayerNorm/betasave_1/RestoreV2:193*
T0*=
_class3
1/loc:@bert/encoder/layer_9/output/LayerNorm/beta*
validate_shape(*
_output_shapes	
:*
use_locking(
í
save_1/Assign_194Assign+bert/encoder/layer_9/output/LayerNorm/gammasave_1/RestoreV2:194*
use_locking(*
T0*>
_class4
20loc:@bert/encoder/layer_9/output/LayerNorm/gamma*
validate_shape(*
_output_shapes	
:
ã
save_1/Assign_195Assign&bert/encoder/layer_9/output/dense/biassave_1/RestoreV2:195*
T0*9
_class/
-+loc:@bert/encoder/layer_9/output/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
ì
save_1/Assign_196Assign(bert/encoder/layer_9/output/dense/kernelsave_1/RestoreV2:196*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*;
_class1
/-loc:@bert/encoder/layer_9/output/dense/kernel
Ã
save_1/Assign_197Assignbert/pooler/dense/biassave_1/RestoreV2:197*
use_locking(*
T0*)
_class
loc:@bert/pooler/dense/bias*
validate_shape(*
_output_shapes	
:
Ì
save_1/Assign_198Assignbert/pooler/dense/kernelsave_1/RestoreV2:198* 
_output_shapes
:
*
use_locking(*
T0*+
_class!
loc:@bert/pooler/dense/kernel*
validate_shape(
¬
save_1/Assign_199Assignoutput_biassave_1/RestoreV2:199*
use_locking(*
T0*
_class
loc:@output_bias*
validate_shape(*
_output_shapes
:
·
save_1/Assign_200Assignoutput_weightssave_1/RestoreV2:200*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*!
_class
loc:@output_weights
à
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_100^save_1/Assign_101^save_1/Assign_102^save_1/Assign_103^save_1/Assign_104^save_1/Assign_105^save_1/Assign_106^save_1/Assign_107^save_1/Assign_108^save_1/Assign_109^save_1/Assign_11^save_1/Assign_110^save_1/Assign_111^save_1/Assign_112^save_1/Assign_113^save_1/Assign_114^save_1/Assign_115^save_1/Assign_116^save_1/Assign_117^save_1/Assign_118^save_1/Assign_119^save_1/Assign_12^save_1/Assign_120^save_1/Assign_121^save_1/Assign_122^save_1/Assign_123^save_1/Assign_124^save_1/Assign_125^save_1/Assign_126^save_1/Assign_127^save_1/Assign_128^save_1/Assign_129^save_1/Assign_13^save_1/Assign_130^save_1/Assign_131^save_1/Assign_132^save_1/Assign_133^save_1/Assign_134^save_1/Assign_135^save_1/Assign_136^save_1/Assign_137^save_1/Assign_138^save_1/Assign_139^save_1/Assign_14^save_1/Assign_140^save_1/Assign_141^save_1/Assign_142^save_1/Assign_143^save_1/Assign_144^save_1/Assign_145^save_1/Assign_146^save_1/Assign_147^save_1/Assign_148^save_1/Assign_149^save_1/Assign_15^save_1/Assign_150^save_1/Assign_151^save_1/Assign_152^save_1/Assign_153^save_1/Assign_154^save_1/Assign_155^save_1/Assign_156^save_1/Assign_157^save_1/Assign_158^save_1/Assign_159^save_1/Assign_16^save_1/Assign_160^save_1/Assign_161^save_1/Assign_162^save_1/Assign_163^save_1/Assign_164^save_1/Assign_165^save_1/Assign_166^save_1/Assign_167^save_1/Assign_168^save_1/Assign_169^save_1/Assign_17^save_1/Assign_170^save_1/Assign_171^save_1/Assign_172^save_1/Assign_173^save_1/Assign_174^save_1/Assign_175^save_1/Assign_176^save_1/Assign_177^save_1/Assign_178^save_1/Assign_179^save_1/Assign_18^save_1/Assign_180^save_1/Assign_181^save_1/Assign_182^save_1/Assign_183^save_1/Assign_184^save_1/Assign_185^save_1/Assign_186^save_1/Assign_187^save_1/Assign_188^save_1/Assign_189^save_1/Assign_19^save_1/Assign_190^save_1/Assign_191^save_1/Assign_192^save_1/Assign_193^save_1/Assign_194^save_1/Assign_195^save_1/Assign_196^save_1/Assign_197^save_1/Assign_198^save_1/Assign_199^save_1/Assign_2^save_1/Assign_20^save_1/Assign_200^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_86^save_1/Assign_87^save_1/Assign_88^save_1/Assign_89^save_1/Assign_9^save_1/Assign_90^save_1/Assign_91^save_1/Assign_92^save_1/Assign_93^save_1/Assign_94^save_1/Assign_95^save_1/Assign_96^save_1/Assign_97^save_1/Assign_98^save_1/Assign_99
1
save_1/restore_allNoOp^save_1/restore_shard "&B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"ï
	variablesýîùî
µ
!bert/embeddings/word_embeddings:0&bert/embeddings/word_embeddings/Assign&bert/embeddings/word_embeddings/read:02>bert/embeddings/word_embeddings/Initializer/truncated_normal:08
Í
'bert/embeddings/token_type_embeddings:0,bert/embeddings/token_type_embeddings/Assign,bert/embeddings/token_type_embeddings/read:02Dbert/embeddings/token_type_embeddings/Initializer/truncated_normal:08
Å
%bert/embeddings/position_embeddings:0*bert/embeddings/position_embeddings/Assign*bert/embeddings/position_embeddings/read:02Bbert/embeddings/position_embeddings/Initializer/truncated_normal:08
¦
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign%bert/embeddings/LayerNorm/beta/read:022bert/embeddings/LayerNorm/beta/Initializer/zeros:08
©
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign&bert/embeddings/LayerNorm/gamma/read:022bert/embeddings/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_0/attention/self/query/kernel:07bert/encoder/layer_0/attention/self/query/kernel/Assign7bert/encoder/layer_0/attention/self/query/kernel/read:02Obert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_0/attention/self/query/bias:05bert/encoder/layer_0/attention/self/query/bias/Assign5bert/encoder/layer_0/attention/self/query/bias/read:02Bbert/encoder/layer_0/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_0/attention/self/key/kernel:05bert/encoder/layer_0/attention/self/key/kernel/Assign5bert/encoder/layer_0/attention/self/key/kernel/read:02Mbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_0/attention/self/key/bias:03bert/encoder/layer_0/attention/self/key/bias/Assign3bert/encoder/layer_0/attention/self/key/bias/read:02@bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_0/attention/self/value/kernel:07bert/encoder/layer_0/attention/self/value/kernel/Assign7bert/encoder/layer_0/attention/self/value/kernel/read:02Obert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_0/attention/self/value/bias:05bert/encoder/layer_0/attention/self/value/bias/Assign5bert/encoder/layer_0/attention/self/value/bias/read:02Bbert/encoder/layer_0/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_0/attention/output/dense/kernel:09bert/encoder/layer_0/attention/output/dense/kernel/Assign9bert/encoder/layer_0/attention/output/dense/kernel/read:02Qbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_0/attention/output/dense/bias:07bert/encoder/layer_0/attention/output/dense/bias/Assign7bert/encoder/layer_0/attention/output/dense/bias/read:02Dbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_0/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_0/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_0/intermediate/dense/kernel:05bert/encoder/layer_0/intermediate/dense/kernel/Assign5bert/encoder/layer_0/intermediate/dense/kernel/read:02Mbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_0/intermediate/dense/bias:03bert/encoder/layer_0/intermediate/dense/bias/Assign3bert/encoder/layer_0/intermediate/dense/bias/read:02@bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_0/output/dense/kernel:0/bert/encoder/layer_0/output/dense/kernel/Assign/bert/encoder/layer_0/output/dense/kernel/read:02Gbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_0/output/dense/bias:0-bert/encoder/layer_0/output/dense/bias/Assign-bert/encoder/layer_0/output/dense/bias/read:02:bert/encoder/layer_0/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign1bert/encoder/layer_0/output/LayerNorm/beta/read:02>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/Assign2bert/encoder/layer_0/output/LayerNorm/gamma/read:02>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_1/attention/self/query/kernel:07bert/encoder/layer_1/attention/self/query/kernel/Assign7bert/encoder/layer_1/attention/self/query/kernel/read:02Obert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_1/attention/self/query/bias:05bert/encoder/layer_1/attention/self/query/bias/Assign5bert/encoder/layer_1/attention/self/query/bias/read:02Bbert/encoder/layer_1/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_1/attention/self/key/kernel:05bert/encoder/layer_1/attention/self/key/kernel/Assign5bert/encoder/layer_1/attention/self/key/kernel/read:02Mbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_1/attention/self/key/bias:03bert/encoder/layer_1/attention/self/key/bias/Assign3bert/encoder/layer_1/attention/self/key/bias/read:02@bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_1/attention/self/value/kernel:07bert/encoder/layer_1/attention/self/value/kernel/Assign7bert/encoder/layer_1/attention/self/value/kernel/read:02Obert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_1/attention/self/value/bias:05bert/encoder/layer_1/attention/self/value/bias/Assign5bert/encoder/layer_1/attention/self/value/bias/read:02Bbert/encoder/layer_1/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_1/attention/output/dense/kernel:09bert/encoder/layer_1/attention/output/dense/kernel/Assign9bert/encoder/layer_1/attention/output/dense/kernel/read:02Qbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_1/attention/output/dense/bias:07bert/encoder/layer_1/attention/output/dense/bias/Assign7bert/encoder/layer_1/attention/output/dense/bias/read:02Dbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_1/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_1/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_1/intermediate/dense/kernel:05bert/encoder/layer_1/intermediate/dense/kernel/Assign5bert/encoder/layer_1/intermediate/dense/kernel/read:02Mbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_1/intermediate/dense/bias:03bert/encoder/layer_1/intermediate/dense/bias/Assign3bert/encoder/layer_1/intermediate/dense/bias/read:02@bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_1/output/dense/kernel:0/bert/encoder/layer_1/output/dense/kernel/Assign/bert/encoder/layer_1/output/dense/kernel/read:02Gbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_1/output/dense/bias:0-bert/encoder/layer_1/output/dense/bias/Assign-bert/encoder/layer_1/output/dense/bias/read:02:bert/encoder/layer_1/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign1bert/encoder/layer_1/output/LayerNorm/beta/read:02>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/Assign2bert/encoder/layer_1/output/LayerNorm/gamma/read:02>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_2/attention/self/query/kernel:07bert/encoder/layer_2/attention/self/query/kernel/Assign7bert/encoder/layer_2/attention/self/query/kernel/read:02Obert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_2/attention/self/query/bias:05bert/encoder/layer_2/attention/self/query/bias/Assign5bert/encoder/layer_2/attention/self/query/bias/read:02Bbert/encoder/layer_2/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_2/attention/self/key/kernel:05bert/encoder/layer_2/attention/self/key/kernel/Assign5bert/encoder/layer_2/attention/self/key/kernel/read:02Mbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_2/attention/self/key/bias:03bert/encoder/layer_2/attention/self/key/bias/Assign3bert/encoder/layer_2/attention/self/key/bias/read:02@bert/encoder/layer_2/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_2/attention/self/value/kernel:07bert/encoder/layer_2/attention/self/value/kernel/Assign7bert/encoder/layer_2/attention/self/value/kernel/read:02Obert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_2/attention/self/value/bias:05bert/encoder/layer_2/attention/self/value/bias/Assign5bert/encoder/layer_2/attention/self/value/bias/read:02Bbert/encoder/layer_2/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_2/attention/output/dense/kernel:09bert/encoder/layer_2/attention/output/dense/kernel/Assign9bert/encoder/layer_2/attention/output/dense/kernel/read:02Qbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_2/attention/output/dense/bias:07bert/encoder/layer_2/attention/output/dense/bias/Assign7bert/encoder/layer_2/attention/output/dense/bias/read:02Dbert/encoder/layer_2/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_2/attention/output/LayerNorm/beta:0;bert/encoder/layer_2/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_2/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_2/attention/output/LayerNorm/gamma:0<bert/encoder/layer_2/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_2/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_2/intermediate/dense/kernel:05bert/encoder/layer_2/intermediate/dense/kernel/Assign5bert/encoder/layer_2/intermediate/dense/kernel/read:02Mbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_2/intermediate/dense/bias:03bert/encoder/layer_2/intermediate/dense/bias/Assign3bert/encoder/layer_2/intermediate/dense/bias/read:02@bert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_2/output/dense/kernel:0/bert/encoder/layer_2/output/dense/kernel/Assign/bert/encoder/layer_2/output/dense/kernel/read:02Gbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_2/output/dense/bias:0-bert/encoder/layer_2/output/dense/bias/Assign-bert/encoder/layer_2/output/dense/bias/read:02:bert/encoder/layer_2/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_2/output/LayerNorm/beta:01bert/encoder/layer_2/output/LayerNorm/beta/Assign1bert/encoder/layer_2/output/LayerNorm/beta/read:02>bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_2/output/LayerNorm/gamma:02bert/encoder/layer_2/output/LayerNorm/gamma/Assign2bert/encoder/layer_2/output/LayerNorm/gamma/read:02>bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_3/attention/self/query/kernel:07bert/encoder/layer_3/attention/self/query/kernel/Assign7bert/encoder/layer_3/attention/self/query/kernel/read:02Obert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_3/attention/self/query/bias:05bert/encoder/layer_3/attention/self/query/bias/Assign5bert/encoder/layer_3/attention/self/query/bias/read:02Bbert/encoder/layer_3/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_3/attention/self/key/kernel:05bert/encoder/layer_3/attention/self/key/kernel/Assign5bert/encoder/layer_3/attention/self/key/kernel/read:02Mbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_3/attention/self/key/bias:03bert/encoder/layer_3/attention/self/key/bias/Assign3bert/encoder/layer_3/attention/self/key/bias/read:02@bert/encoder/layer_3/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_3/attention/self/value/kernel:07bert/encoder/layer_3/attention/self/value/kernel/Assign7bert/encoder/layer_3/attention/self/value/kernel/read:02Obert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_3/attention/self/value/bias:05bert/encoder/layer_3/attention/self/value/bias/Assign5bert/encoder/layer_3/attention/self/value/bias/read:02Bbert/encoder/layer_3/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_3/attention/output/dense/kernel:09bert/encoder/layer_3/attention/output/dense/kernel/Assign9bert/encoder/layer_3/attention/output/dense/kernel/read:02Qbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_3/attention/output/dense/bias:07bert/encoder/layer_3/attention/output/dense/bias/Assign7bert/encoder/layer_3/attention/output/dense/bias/read:02Dbert/encoder/layer_3/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_3/attention/output/LayerNorm/beta:0;bert/encoder/layer_3/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_3/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_3/attention/output/LayerNorm/gamma:0<bert/encoder/layer_3/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_3/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_3/intermediate/dense/kernel:05bert/encoder/layer_3/intermediate/dense/kernel/Assign5bert/encoder/layer_3/intermediate/dense/kernel/read:02Mbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_3/intermediate/dense/bias:03bert/encoder/layer_3/intermediate/dense/bias/Assign3bert/encoder/layer_3/intermediate/dense/bias/read:02@bert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_3/output/dense/kernel:0/bert/encoder/layer_3/output/dense/kernel/Assign/bert/encoder/layer_3/output/dense/kernel/read:02Gbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_3/output/dense/bias:0-bert/encoder/layer_3/output/dense/bias/Assign-bert/encoder/layer_3/output/dense/bias/read:02:bert/encoder/layer_3/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_3/output/LayerNorm/beta:01bert/encoder/layer_3/output/LayerNorm/beta/Assign1bert/encoder/layer_3/output/LayerNorm/beta/read:02>bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_3/output/LayerNorm/gamma:02bert/encoder/layer_3/output/LayerNorm/gamma/Assign2bert/encoder/layer_3/output/LayerNorm/gamma/read:02>bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_4/attention/self/query/kernel:07bert/encoder/layer_4/attention/self/query/kernel/Assign7bert/encoder/layer_4/attention/self/query/kernel/read:02Obert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_4/attention/self/query/bias:05bert/encoder/layer_4/attention/self/query/bias/Assign5bert/encoder/layer_4/attention/self/query/bias/read:02Bbert/encoder/layer_4/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_4/attention/self/key/kernel:05bert/encoder/layer_4/attention/self/key/kernel/Assign5bert/encoder/layer_4/attention/self/key/kernel/read:02Mbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_4/attention/self/key/bias:03bert/encoder/layer_4/attention/self/key/bias/Assign3bert/encoder/layer_4/attention/self/key/bias/read:02@bert/encoder/layer_4/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_4/attention/self/value/kernel:07bert/encoder/layer_4/attention/self/value/kernel/Assign7bert/encoder/layer_4/attention/self/value/kernel/read:02Obert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_4/attention/self/value/bias:05bert/encoder/layer_4/attention/self/value/bias/Assign5bert/encoder/layer_4/attention/self/value/bias/read:02Bbert/encoder/layer_4/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_4/attention/output/dense/kernel:09bert/encoder/layer_4/attention/output/dense/kernel/Assign9bert/encoder/layer_4/attention/output/dense/kernel/read:02Qbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_4/attention/output/dense/bias:07bert/encoder/layer_4/attention/output/dense/bias/Assign7bert/encoder/layer_4/attention/output/dense/bias/read:02Dbert/encoder/layer_4/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_4/attention/output/LayerNorm/beta:0;bert/encoder/layer_4/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_4/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_4/attention/output/LayerNorm/gamma:0<bert/encoder/layer_4/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_4/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_4/intermediate/dense/kernel:05bert/encoder/layer_4/intermediate/dense/kernel/Assign5bert/encoder/layer_4/intermediate/dense/kernel/read:02Mbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_4/intermediate/dense/bias:03bert/encoder/layer_4/intermediate/dense/bias/Assign3bert/encoder/layer_4/intermediate/dense/bias/read:02@bert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_4/output/dense/kernel:0/bert/encoder/layer_4/output/dense/kernel/Assign/bert/encoder/layer_4/output/dense/kernel/read:02Gbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_4/output/dense/bias:0-bert/encoder/layer_4/output/dense/bias/Assign-bert/encoder/layer_4/output/dense/bias/read:02:bert/encoder/layer_4/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_4/output/LayerNorm/beta:01bert/encoder/layer_4/output/LayerNorm/beta/Assign1bert/encoder/layer_4/output/LayerNorm/beta/read:02>bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_4/output/LayerNorm/gamma:02bert/encoder/layer_4/output/LayerNorm/gamma/Assign2bert/encoder/layer_4/output/LayerNorm/gamma/read:02>bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_5/attention/self/query/kernel:07bert/encoder/layer_5/attention/self/query/kernel/Assign7bert/encoder/layer_5/attention/self/query/kernel/read:02Obert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_5/attention/self/query/bias:05bert/encoder/layer_5/attention/self/query/bias/Assign5bert/encoder/layer_5/attention/self/query/bias/read:02Bbert/encoder/layer_5/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_5/attention/self/key/kernel:05bert/encoder/layer_5/attention/self/key/kernel/Assign5bert/encoder/layer_5/attention/self/key/kernel/read:02Mbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_5/attention/self/key/bias:03bert/encoder/layer_5/attention/self/key/bias/Assign3bert/encoder/layer_5/attention/self/key/bias/read:02@bert/encoder/layer_5/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_5/attention/self/value/kernel:07bert/encoder/layer_5/attention/self/value/kernel/Assign7bert/encoder/layer_5/attention/self/value/kernel/read:02Obert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_5/attention/self/value/bias:05bert/encoder/layer_5/attention/self/value/bias/Assign5bert/encoder/layer_5/attention/self/value/bias/read:02Bbert/encoder/layer_5/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_5/attention/output/dense/kernel:09bert/encoder/layer_5/attention/output/dense/kernel/Assign9bert/encoder/layer_5/attention/output/dense/kernel/read:02Qbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_5/attention/output/dense/bias:07bert/encoder/layer_5/attention/output/dense/bias/Assign7bert/encoder/layer_5/attention/output/dense/bias/read:02Dbert/encoder/layer_5/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_5/attention/output/LayerNorm/beta:0;bert/encoder/layer_5/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_5/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_5/attention/output/LayerNorm/gamma:0<bert/encoder/layer_5/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_5/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_5/intermediate/dense/kernel:05bert/encoder/layer_5/intermediate/dense/kernel/Assign5bert/encoder/layer_5/intermediate/dense/kernel/read:02Mbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_5/intermediate/dense/bias:03bert/encoder/layer_5/intermediate/dense/bias/Assign3bert/encoder/layer_5/intermediate/dense/bias/read:02@bert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_5/output/dense/kernel:0/bert/encoder/layer_5/output/dense/kernel/Assign/bert/encoder/layer_5/output/dense/kernel/read:02Gbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_5/output/dense/bias:0-bert/encoder/layer_5/output/dense/bias/Assign-bert/encoder/layer_5/output/dense/bias/read:02:bert/encoder/layer_5/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_5/output/LayerNorm/beta:01bert/encoder/layer_5/output/LayerNorm/beta/Assign1bert/encoder/layer_5/output/LayerNorm/beta/read:02>bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_5/output/LayerNorm/gamma:02bert/encoder/layer_5/output/LayerNorm/gamma/Assign2bert/encoder/layer_5/output/LayerNorm/gamma/read:02>bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_6/attention/self/query/kernel:07bert/encoder/layer_6/attention/self/query/kernel/Assign7bert/encoder/layer_6/attention/self/query/kernel/read:02Obert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_6/attention/self/query/bias:05bert/encoder/layer_6/attention/self/query/bias/Assign5bert/encoder/layer_6/attention/self/query/bias/read:02Bbert/encoder/layer_6/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_6/attention/self/key/kernel:05bert/encoder/layer_6/attention/self/key/kernel/Assign5bert/encoder/layer_6/attention/self/key/kernel/read:02Mbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_6/attention/self/key/bias:03bert/encoder/layer_6/attention/self/key/bias/Assign3bert/encoder/layer_6/attention/self/key/bias/read:02@bert/encoder/layer_6/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_6/attention/self/value/kernel:07bert/encoder/layer_6/attention/self/value/kernel/Assign7bert/encoder/layer_6/attention/self/value/kernel/read:02Obert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_6/attention/self/value/bias:05bert/encoder/layer_6/attention/self/value/bias/Assign5bert/encoder/layer_6/attention/self/value/bias/read:02Bbert/encoder/layer_6/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_6/attention/output/dense/kernel:09bert/encoder/layer_6/attention/output/dense/kernel/Assign9bert/encoder/layer_6/attention/output/dense/kernel/read:02Qbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_6/attention/output/dense/bias:07bert/encoder/layer_6/attention/output/dense/bias/Assign7bert/encoder/layer_6/attention/output/dense/bias/read:02Dbert/encoder/layer_6/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_6/attention/output/LayerNorm/beta:0;bert/encoder/layer_6/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_6/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_6/attention/output/LayerNorm/gamma:0<bert/encoder/layer_6/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_6/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_6/intermediate/dense/kernel:05bert/encoder/layer_6/intermediate/dense/kernel/Assign5bert/encoder/layer_6/intermediate/dense/kernel/read:02Mbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_6/intermediate/dense/bias:03bert/encoder/layer_6/intermediate/dense/bias/Assign3bert/encoder/layer_6/intermediate/dense/bias/read:02@bert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_6/output/dense/kernel:0/bert/encoder/layer_6/output/dense/kernel/Assign/bert/encoder/layer_6/output/dense/kernel/read:02Gbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_6/output/dense/bias:0-bert/encoder/layer_6/output/dense/bias/Assign-bert/encoder/layer_6/output/dense/bias/read:02:bert/encoder/layer_6/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_6/output/LayerNorm/beta:01bert/encoder/layer_6/output/LayerNorm/beta/Assign1bert/encoder/layer_6/output/LayerNorm/beta/read:02>bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_6/output/LayerNorm/gamma:02bert/encoder/layer_6/output/LayerNorm/gamma/Assign2bert/encoder/layer_6/output/LayerNorm/gamma/read:02>bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_7/attention/self/query/kernel:07bert/encoder/layer_7/attention/self/query/kernel/Assign7bert/encoder/layer_7/attention/self/query/kernel/read:02Obert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_7/attention/self/query/bias:05bert/encoder/layer_7/attention/self/query/bias/Assign5bert/encoder/layer_7/attention/self/query/bias/read:02Bbert/encoder/layer_7/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_7/attention/self/key/kernel:05bert/encoder/layer_7/attention/self/key/kernel/Assign5bert/encoder/layer_7/attention/self/key/kernel/read:02Mbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_7/attention/self/key/bias:03bert/encoder/layer_7/attention/self/key/bias/Assign3bert/encoder/layer_7/attention/self/key/bias/read:02@bert/encoder/layer_7/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_7/attention/self/value/kernel:07bert/encoder/layer_7/attention/self/value/kernel/Assign7bert/encoder/layer_7/attention/self/value/kernel/read:02Obert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_7/attention/self/value/bias:05bert/encoder/layer_7/attention/self/value/bias/Assign5bert/encoder/layer_7/attention/self/value/bias/read:02Bbert/encoder/layer_7/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_7/attention/output/dense/kernel:09bert/encoder/layer_7/attention/output/dense/kernel/Assign9bert/encoder/layer_7/attention/output/dense/kernel/read:02Qbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_7/attention/output/dense/bias:07bert/encoder/layer_7/attention/output/dense/bias/Assign7bert/encoder/layer_7/attention/output/dense/bias/read:02Dbert/encoder/layer_7/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_7/attention/output/LayerNorm/beta:0;bert/encoder/layer_7/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_7/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_7/attention/output/LayerNorm/gamma:0<bert/encoder/layer_7/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_7/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_7/intermediate/dense/kernel:05bert/encoder/layer_7/intermediate/dense/kernel/Assign5bert/encoder/layer_7/intermediate/dense/kernel/read:02Mbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_7/intermediate/dense/bias:03bert/encoder/layer_7/intermediate/dense/bias/Assign3bert/encoder/layer_7/intermediate/dense/bias/read:02@bert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_7/output/dense/kernel:0/bert/encoder/layer_7/output/dense/kernel/Assign/bert/encoder/layer_7/output/dense/kernel/read:02Gbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_7/output/dense/bias:0-bert/encoder/layer_7/output/dense/bias/Assign-bert/encoder/layer_7/output/dense/bias/read:02:bert/encoder/layer_7/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_7/output/LayerNorm/beta:01bert/encoder/layer_7/output/LayerNorm/beta/Assign1bert/encoder/layer_7/output/LayerNorm/beta/read:02>bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_7/output/LayerNorm/gamma:02bert/encoder/layer_7/output/LayerNorm/gamma/Assign2bert/encoder/layer_7/output/LayerNorm/gamma/read:02>bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_8/attention/self/query/kernel:07bert/encoder/layer_8/attention/self/query/kernel/Assign7bert/encoder/layer_8/attention/self/query/kernel/read:02Obert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_8/attention/self/query/bias:05bert/encoder/layer_8/attention/self/query/bias/Assign5bert/encoder/layer_8/attention/self/query/bias/read:02Bbert/encoder/layer_8/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_8/attention/self/key/kernel:05bert/encoder/layer_8/attention/self/key/kernel/Assign5bert/encoder/layer_8/attention/self/key/kernel/read:02Mbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_8/attention/self/key/bias:03bert/encoder/layer_8/attention/self/key/bias/Assign3bert/encoder/layer_8/attention/self/key/bias/read:02@bert/encoder/layer_8/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_8/attention/self/value/kernel:07bert/encoder/layer_8/attention/self/value/kernel/Assign7bert/encoder/layer_8/attention/self/value/kernel/read:02Obert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_8/attention/self/value/bias:05bert/encoder/layer_8/attention/self/value/bias/Assign5bert/encoder/layer_8/attention/self/value/bias/read:02Bbert/encoder/layer_8/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_8/attention/output/dense/kernel:09bert/encoder/layer_8/attention/output/dense/kernel/Assign9bert/encoder/layer_8/attention/output/dense/kernel/read:02Qbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_8/attention/output/dense/bias:07bert/encoder/layer_8/attention/output/dense/bias/Assign7bert/encoder/layer_8/attention/output/dense/bias/read:02Dbert/encoder/layer_8/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_8/attention/output/LayerNorm/beta:0;bert/encoder/layer_8/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_8/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_8/attention/output/LayerNorm/gamma:0<bert/encoder/layer_8/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_8/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_8/intermediate/dense/kernel:05bert/encoder/layer_8/intermediate/dense/kernel/Assign5bert/encoder/layer_8/intermediate/dense/kernel/read:02Mbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_8/intermediate/dense/bias:03bert/encoder/layer_8/intermediate/dense/bias/Assign3bert/encoder/layer_8/intermediate/dense/bias/read:02@bert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_8/output/dense/kernel:0/bert/encoder/layer_8/output/dense/kernel/Assign/bert/encoder/layer_8/output/dense/kernel/read:02Gbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_8/output/dense/bias:0-bert/encoder/layer_8/output/dense/bias/Assign-bert/encoder/layer_8/output/dense/bias/read:02:bert/encoder/layer_8/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_8/output/LayerNorm/beta:01bert/encoder/layer_8/output/LayerNorm/beta/Assign1bert/encoder/layer_8/output/LayerNorm/beta/read:02>bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_8/output/LayerNorm/gamma:02bert/encoder/layer_8/output/LayerNorm/gamma/Assign2bert/encoder/layer_8/output/LayerNorm/gamma/read:02>bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_9/attention/self/query/kernel:07bert/encoder/layer_9/attention/self/query/kernel/Assign7bert/encoder/layer_9/attention/self/query/kernel/read:02Obert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_9/attention/self/query/bias:05bert/encoder/layer_9/attention/self/query/bias/Assign5bert/encoder/layer_9/attention/self/query/bias/read:02Bbert/encoder/layer_9/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_9/attention/self/key/kernel:05bert/encoder/layer_9/attention/self/key/kernel/Assign5bert/encoder/layer_9/attention/self/key/kernel/read:02Mbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_9/attention/self/key/bias:03bert/encoder/layer_9/attention/self/key/bias/Assign3bert/encoder/layer_9/attention/self/key/bias/read:02@bert/encoder/layer_9/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_9/attention/self/value/kernel:07bert/encoder/layer_9/attention/self/value/kernel/Assign7bert/encoder/layer_9/attention/self/value/kernel/read:02Obert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_9/attention/self/value/bias:05bert/encoder/layer_9/attention/self/value/bias/Assign5bert/encoder/layer_9/attention/self/value/bias/read:02Bbert/encoder/layer_9/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_9/attention/output/dense/kernel:09bert/encoder/layer_9/attention/output/dense/kernel/Assign9bert/encoder/layer_9/attention/output/dense/kernel/read:02Qbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_9/attention/output/dense/bias:07bert/encoder/layer_9/attention/output/dense/bias/Assign7bert/encoder/layer_9/attention/output/dense/bias/read:02Dbert/encoder/layer_9/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_9/attention/output/LayerNorm/beta:0;bert/encoder/layer_9/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_9/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_9/attention/output/LayerNorm/gamma:0<bert/encoder/layer_9/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_9/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_9/intermediate/dense/kernel:05bert/encoder/layer_9/intermediate/dense/kernel/Assign5bert/encoder/layer_9/intermediate/dense/kernel/read:02Mbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_9/intermediate/dense/bias:03bert/encoder/layer_9/intermediate/dense/bias/Assign3bert/encoder/layer_9/intermediate/dense/bias/read:02@bert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_9/output/dense/kernel:0/bert/encoder/layer_9/output/dense/kernel/Assign/bert/encoder/layer_9/output/dense/kernel/read:02Gbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_9/output/dense/bias:0-bert/encoder/layer_9/output/dense/bias/Assign-bert/encoder/layer_9/output/dense/bias/read:02:bert/encoder/layer_9/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_9/output/LayerNorm/beta:01bert/encoder/layer_9/output/LayerNorm/beta/Assign1bert/encoder/layer_9/output/LayerNorm/beta/read:02>bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_9/output/LayerNorm/gamma:02bert/encoder/layer_9/output/LayerNorm/gamma/Assign2bert/encoder/layer_9/output/LayerNorm/gamma/read:02>bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/ones:08
ý
3bert/encoder/layer_10/attention/self/query/kernel:08bert/encoder/layer_10/attention/self/query/kernel/Assign8bert/encoder/layer_10/attention/self/query/kernel/read:02Pbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal:08
ê
1bert/encoder/layer_10/attention/self/query/bias:06bert/encoder/layer_10/attention/self/query/bias/Assign6bert/encoder/layer_10/attention/self/query/bias/read:02Cbert/encoder/layer_10/attention/self/query/bias/Initializer/zeros:08
õ
1bert/encoder/layer_10/attention/self/key/kernel:06bert/encoder/layer_10/attention/self/key/kernel/Assign6bert/encoder/layer_10/attention/self/key/kernel/read:02Nbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal:08
â
/bert/encoder/layer_10/attention/self/key/bias:04bert/encoder/layer_10/attention/self/key/bias/Assign4bert/encoder/layer_10/attention/self/key/bias/read:02Abert/encoder/layer_10/attention/self/key/bias/Initializer/zeros:08
ý
3bert/encoder/layer_10/attention/self/value/kernel:08bert/encoder/layer_10/attention/self/value/kernel/Assign8bert/encoder/layer_10/attention/self/value/kernel/read:02Pbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal:08
ê
1bert/encoder/layer_10/attention/self/value/bias:06bert/encoder/layer_10/attention/self/value/bias/Assign6bert/encoder/layer_10/attention/self/value/bias/read:02Cbert/encoder/layer_10/attention/self/value/bias/Initializer/zeros:08

5bert/encoder/layer_10/attention/output/dense/kernel:0:bert/encoder/layer_10/attention/output/dense/kernel/Assign:bert/encoder/layer_10/attention/output/dense/kernel/read:02Rbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal:08
ò
3bert/encoder/layer_10/attention/output/dense/bias:08bert/encoder/layer_10/attention/output/dense/bias/Assign8bert/encoder/layer_10/attention/output/dense/bias/read:02Ebert/encoder/layer_10/attention/output/dense/bias/Initializer/zeros:08

7bert/encoder/layer_10/attention/output/LayerNorm/beta:0<bert/encoder/layer_10/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_10/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zeros:08

8bert/encoder/layer_10/attention/output/LayerNorm/gamma:0=bert/encoder/layer_10/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_10/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/ones:08
õ
1bert/encoder/layer_10/intermediate/dense/kernel:06bert/encoder/layer_10/intermediate/dense/kernel/Assign6bert/encoder/layer_10/intermediate/dense/kernel/read:02Nbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal:08
â
/bert/encoder/layer_10/intermediate/dense/bias:04bert/encoder/layer_10/intermediate/dense/bias/Assign4bert/encoder/layer_10/intermediate/dense/bias/read:02Abert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros:08
Ý
+bert/encoder/layer_10/output/dense/kernel:00bert/encoder/layer_10/output/dense/kernel/Assign0bert/encoder/layer_10/output/dense/kernel/read:02Hbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal:08
Ê
)bert/encoder/layer_10/output/dense/bias:0.bert/encoder/layer_10/output/dense/bias/Assign.bert/encoder/layer_10/output/dense/bias/read:02;bert/encoder/layer_10/output/dense/bias/Initializer/zeros:08
Ú
-bert/encoder/layer_10/output/LayerNorm/beta:02bert/encoder/layer_10/output/LayerNorm/beta/Assign2bert/encoder/layer_10/output/LayerNorm/beta/read:02?bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zeros:08
Ý
.bert/encoder/layer_10/output/LayerNorm/gamma:03bert/encoder/layer_10/output/LayerNorm/gamma/Assign3bert/encoder/layer_10/output/LayerNorm/gamma/read:02?bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/ones:08
ý
3bert/encoder/layer_11/attention/self/query/kernel:08bert/encoder/layer_11/attention/self/query/kernel/Assign8bert/encoder/layer_11/attention/self/query/kernel/read:02Pbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal:08
ê
1bert/encoder/layer_11/attention/self/query/bias:06bert/encoder/layer_11/attention/self/query/bias/Assign6bert/encoder/layer_11/attention/self/query/bias/read:02Cbert/encoder/layer_11/attention/self/query/bias/Initializer/zeros:08
õ
1bert/encoder/layer_11/attention/self/key/kernel:06bert/encoder/layer_11/attention/self/key/kernel/Assign6bert/encoder/layer_11/attention/self/key/kernel/read:02Nbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal:08
â
/bert/encoder/layer_11/attention/self/key/bias:04bert/encoder/layer_11/attention/self/key/bias/Assign4bert/encoder/layer_11/attention/self/key/bias/read:02Abert/encoder/layer_11/attention/self/key/bias/Initializer/zeros:08
ý
3bert/encoder/layer_11/attention/self/value/kernel:08bert/encoder/layer_11/attention/self/value/kernel/Assign8bert/encoder/layer_11/attention/self/value/kernel/read:02Pbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal:08
ê
1bert/encoder/layer_11/attention/self/value/bias:06bert/encoder/layer_11/attention/self/value/bias/Assign6bert/encoder/layer_11/attention/self/value/bias/read:02Cbert/encoder/layer_11/attention/self/value/bias/Initializer/zeros:08

5bert/encoder/layer_11/attention/output/dense/kernel:0:bert/encoder/layer_11/attention/output/dense/kernel/Assign:bert/encoder/layer_11/attention/output/dense/kernel/read:02Rbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal:08
ò
3bert/encoder/layer_11/attention/output/dense/bias:08bert/encoder/layer_11/attention/output/dense/bias/Assign8bert/encoder/layer_11/attention/output/dense/bias/read:02Ebert/encoder/layer_11/attention/output/dense/bias/Initializer/zeros:08

7bert/encoder/layer_11/attention/output/LayerNorm/beta:0<bert/encoder/layer_11/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_11/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zeros:08

8bert/encoder/layer_11/attention/output/LayerNorm/gamma:0=bert/encoder/layer_11/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_11/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/ones:08
õ
1bert/encoder/layer_11/intermediate/dense/kernel:06bert/encoder/layer_11/intermediate/dense/kernel/Assign6bert/encoder/layer_11/intermediate/dense/kernel/read:02Nbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal:08
â
/bert/encoder/layer_11/intermediate/dense/bias:04bert/encoder/layer_11/intermediate/dense/bias/Assign4bert/encoder/layer_11/intermediate/dense/bias/read:02Abert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros:08
Ý
+bert/encoder/layer_11/output/dense/kernel:00bert/encoder/layer_11/output/dense/kernel/Assign0bert/encoder/layer_11/output/dense/kernel/read:02Hbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal:08
Ê
)bert/encoder/layer_11/output/dense/bias:0.bert/encoder/layer_11/output/dense/bias/Assign.bert/encoder/layer_11/output/dense/bias/read:02;bert/encoder/layer_11/output/dense/bias/Initializer/zeros:08
Ú
-bert/encoder/layer_11/output/LayerNorm/beta:02bert/encoder/layer_11/output/LayerNorm/beta/Assign2bert/encoder/layer_11/output/LayerNorm/beta/read:02?bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zeros:08
Ý
.bert/encoder/layer_11/output/LayerNorm/gamma:03bert/encoder/layer_11/output/LayerNorm/gamma/Assign3bert/encoder/layer_11/output/LayerNorm/gamma/read:02?bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/ones:08

bert/pooler/dense/kernel:0bert/pooler/dense/kernel/Assignbert/pooler/dense/kernel/read:027bert/pooler/dense/kernel/Initializer/truncated_normal:08

bert/pooler/dense/bias:0bert/pooler/dense/bias/Assignbert/pooler/dense/bias/read:02*bert/pooler/dense/bias/Initializer/zeros:08
q
output_weights:0output_weights/Assignoutput_weights/read:02-output_weights/Initializer/truncated_normal:08
Z
output_bias:0output_bias/Assignoutput_bias/read:02output_bias/Initializer/zeros:08"Ä\
model_variables°\­\
¦
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign%bert/embeddings/LayerNorm/beta/read:022bert/embeddings/LayerNorm/beta/Initializer/zeros:08
©
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign&bert/embeddings/LayerNorm/gamma/read:022bert/embeddings/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_0/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_0/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign1bert/encoder/layer_0/output/LayerNorm/beta/read:02>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/Assign2bert/encoder/layer_0/output/LayerNorm/gamma/read:02>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_1/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_1/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign1bert/encoder/layer_1/output/LayerNorm/beta/read:02>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/Assign2bert/encoder/layer_1/output/LayerNorm/gamma/read:02>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_2/attention/output/LayerNorm/beta:0;bert/encoder/layer_2/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_2/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_2/attention/output/LayerNorm/gamma:0<bert/encoder/layer_2/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_2/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_2/output/LayerNorm/beta:01bert/encoder/layer_2/output/LayerNorm/beta/Assign1bert/encoder/layer_2/output/LayerNorm/beta/read:02>bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_2/output/LayerNorm/gamma:02bert/encoder/layer_2/output/LayerNorm/gamma/Assign2bert/encoder/layer_2/output/LayerNorm/gamma/read:02>bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_3/attention/output/LayerNorm/beta:0;bert/encoder/layer_3/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_3/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_3/attention/output/LayerNorm/gamma:0<bert/encoder/layer_3/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_3/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_3/output/LayerNorm/beta:01bert/encoder/layer_3/output/LayerNorm/beta/Assign1bert/encoder/layer_3/output/LayerNorm/beta/read:02>bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_3/output/LayerNorm/gamma:02bert/encoder/layer_3/output/LayerNorm/gamma/Assign2bert/encoder/layer_3/output/LayerNorm/gamma/read:02>bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_4/attention/output/LayerNorm/beta:0;bert/encoder/layer_4/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_4/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_4/attention/output/LayerNorm/gamma:0<bert/encoder/layer_4/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_4/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_4/output/LayerNorm/beta:01bert/encoder/layer_4/output/LayerNorm/beta/Assign1bert/encoder/layer_4/output/LayerNorm/beta/read:02>bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_4/output/LayerNorm/gamma:02bert/encoder/layer_4/output/LayerNorm/gamma/Assign2bert/encoder/layer_4/output/LayerNorm/gamma/read:02>bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_5/attention/output/LayerNorm/beta:0;bert/encoder/layer_5/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_5/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_5/attention/output/LayerNorm/gamma:0<bert/encoder/layer_5/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_5/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_5/output/LayerNorm/beta:01bert/encoder/layer_5/output/LayerNorm/beta/Assign1bert/encoder/layer_5/output/LayerNorm/beta/read:02>bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_5/output/LayerNorm/gamma:02bert/encoder/layer_5/output/LayerNorm/gamma/Assign2bert/encoder/layer_5/output/LayerNorm/gamma/read:02>bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_6/attention/output/LayerNorm/beta:0;bert/encoder/layer_6/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_6/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_6/attention/output/LayerNorm/gamma:0<bert/encoder/layer_6/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_6/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_6/output/LayerNorm/beta:01bert/encoder/layer_6/output/LayerNorm/beta/Assign1bert/encoder/layer_6/output/LayerNorm/beta/read:02>bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_6/output/LayerNorm/gamma:02bert/encoder/layer_6/output/LayerNorm/gamma/Assign2bert/encoder/layer_6/output/LayerNorm/gamma/read:02>bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_7/attention/output/LayerNorm/beta:0;bert/encoder/layer_7/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_7/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_7/attention/output/LayerNorm/gamma:0<bert/encoder/layer_7/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_7/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_7/output/LayerNorm/beta:01bert/encoder/layer_7/output/LayerNorm/beta/Assign1bert/encoder/layer_7/output/LayerNorm/beta/read:02>bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_7/output/LayerNorm/gamma:02bert/encoder/layer_7/output/LayerNorm/gamma/Assign2bert/encoder/layer_7/output/LayerNorm/gamma/read:02>bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_8/attention/output/LayerNorm/beta:0;bert/encoder/layer_8/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_8/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_8/attention/output/LayerNorm/gamma:0<bert/encoder/layer_8/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_8/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_8/output/LayerNorm/beta:01bert/encoder/layer_8/output/LayerNorm/beta/Assign1bert/encoder/layer_8/output/LayerNorm/beta/read:02>bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_8/output/LayerNorm/gamma:02bert/encoder/layer_8/output/LayerNorm/gamma/Assign2bert/encoder/layer_8/output/LayerNorm/gamma/read:02>bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/ones:08
þ
6bert/encoder/layer_9/attention/output/LayerNorm/beta:0;bert/encoder/layer_9/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_9/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_9/attention/output/LayerNorm/gamma:0<bert/encoder/layer_9/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_9/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/ones:08
Ö
,bert/encoder/layer_9/output/LayerNorm/beta:01bert/encoder/layer_9/output/LayerNorm/beta/Assign1bert/encoder/layer_9/output/LayerNorm/beta/read:02>bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_9/output/LayerNorm/gamma:02bert/encoder/layer_9/output/LayerNorm/gamma/Assign2bert/encoder/layer_9/output/LayerNorm/gamma/read:02>bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/ones:08

7bert/encoder/layer_10/attention/output/LayerNorm/beta:0<bert/encoder/layer_10/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_10/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zeros:08

8bert/encoder/layer_10/attention/output/LayerNorm/gamma:0=bert/encoder/layer_10/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_10/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/ones:08
Ú
-bert/encoder/layer_10/output/LayerNorm/beta:02bert/encoder/layer_10/output/LayerNorm/beta/Assign2bert/encoder/layer_10/output/LayerNorm/beta/read:02?bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zeros:08
Ý
.bert/encoder/layer_10/output/LayerNorm/gamma:03bert/encoder/layer_10/output/LayerNorm/gamma/Assign3bert/encoder/layer_10/output/LayerNorm/gamma/read:02?bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/ones:08

7bert/encoder/layer_11/attention/output/LayerNorm/beta:0<bert/encoder/layer_11/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_11/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zeros:08

8bert/encoder/layer_11/attention/output/LayerNorm/gamma:0=bert/encoder/layer_11/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_11/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/ones:08
Ú
-bert/encoder/layer_11/output/LayerNorm/beta:02bert/encoder/layer_11/output/LayerNorm/beta/Assign2bert/encoder/layer_11/output/LayerNorm/beta/read:02?bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zeros:08
Ý
.bert/encoder/layer_11/output/LayerNorm/gamma:03bert/encoder/layer_11/output/LayerNorm/gamma/Assign3bert/encoder/layer_11/output/LayerNorm/gamma/read:02?bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/ones:08"ï
trainable_variablesýîùî
µ
!bert/embeddings/word_embeddings:0&bert/embeddings/word_embeddings/Assign&bert/embeddings/word_embeddings/read:02>bert/embeddings/word_embeddings/Initializer/truncated_normal:08
Í
'bert/embeddings/token_type_embeddings:0,bert/embeddings/token_type_embeddings/Assign,bert/embeddings/token_type_embeddings/read:02Dbert/embeddings/token_type_embeddings/Initializer/truncated_normal:08
Å
%bert/embeddings/position_embeddings:0*bert/embeddings/position_embeddings/Assign*bert/embeddings/position_embeddings/read:02Bbert/embeddings/position_embeddings/Initializer/truncated_normal:08
¦
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign%bert/embeddings/LayerNorm/beta/read:022bert/embeddings/LayerNorm/beta/Initializer/zeros:08
©
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign&bert/embeddings/LayerNorm/gamma/read:022bert/embeddings/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_0/attention/self/query/kernel:07bert/encoder/layer_0/attention/self/query/kernel/Assign7bert/encoder/layer_0/attention/self/query/kernel/read:02Obert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_0/attention/self/query/bias:05bert/encoder/layer_0/attention/self/query/bias/Assign5bert/encoder/layer_0/attention/self/query/bias/read:02Bbert/encoder/layer_0/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_0/attention/self/key/kernel:05bert/encoder/layer_0/attention/self/key/kernel/Assign5bert/encoder/layer_0/attention/self/key/kernel/read:02Mbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_0/attention/self/key/bias:03bert/encoder/layer_0/attention/self/key/bias/Assign3bert/encoder/layer_0/attention/self/key/bias/read:02@bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_0/attention/self/value/kernel:07bert/encoder/layer_0/attention/self/value/kernel/Assign7bert/encoder/layer_0/attention/self/value/kernel/read:02Obert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_0/attention/self/value/bias:05bert/encoder/layer_0/attention/self/value/bias/Assign5bert/encoder/layer_0/attention/self/value/bias/read:02Bbert/encoder/layer_0/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_0/attention/output/dense/kernel:09bert/encoder/layer_0/attention/output/dense/kernel/Assign9bert/encoder/layer_0/attention/output/dense/kernel/read:02Qbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_0/attention/output/dense/bias:07bert/encoder/layer_0/attention/output/dense/bias/Assign7bert/encoder/layer_0/attention/output/dense/bias/read:02Dbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_0/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_0/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_0/intermediate/dense/kernel:05bert/encoder/layer_0/intermediate/dense/kernel/Assign5bert/encoder/layer_0/intermediate/dense/kernel/read:02Mbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_0/intermediate/dense/bias:03bert/encoder/layer_0/intermediate/dense/bias/Assign3bert/encoder/layer_0/intermediate/dense/bias/read:02@bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_0/output/dense/kernel:0/bert/encoder/layer_0/output/dense/kernel/Assign/bert/encoder/layer_0/output/dense/kernel/read:02Gbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_0/output/dense/bias:0-bert/encoder/layer_0/output/dense/bias/Assign-bert/encoder/layer_0/output/dense/bias/read:02:bert/encoder/layer_0/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign1bert/encoder/layer_0/output/LayerNorm/beta/read:02>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/Assign2bert/encoder/layer_0/output/LayerNorm/gamma/read:02>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_1/attention/self/query/kernel:07bert/encoder/layer_1/attention/self/query/kernel/Assign7bert/encoder/layer_1/attention/self/query/kernel/read:02Obert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_1/attention/self/query/bias:05bert/encoder/layer_1/attention/self/query/bias/Assign5bert/encoder/layer_1/attention/self/query/bias/read:02Bbert/encoder/layer_1/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_1/attention/self/key/kernel:05bert/encoder/layer_1/attention/self/key/kernel/Assign5bert/encoder/layer_1/attention/self/key/kernel/read:02Mbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_1/attention/self/key/bias:03bert/encoder/layer_1/attention/self/key/bias/Assign3bert/encoder/layer_1/attention/self/key/bias/read:02@bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_1/attention/self/value/kernel:07bert/encoder/layer_1/attention/self/value/kernel/Assign7bert/encoder/layer_1/attention/self/value/kernel/read:02Obert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_1/attention/self/value/bias:05bert/encoder/layer_1/attention/self/value/bias/Assign5bert/encoder/layer_1/attention/self/value/bias/read:02Bbert/encoder/layer_1/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_1/attention/output/dense/kernel:09bert/encoder/layer_1/attention/output/dense/kernel/Assign9bert/encoder/layer_1/attention/output/dense/kernel/read:02Qbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_1/attention/output/dense/bias:07bert/encoder/layer_1/attention/output/dense/bias/Assign7bert/encoder/layer_1/attention/output/dense/bias/read:02Dbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_1/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_1/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_1/intermediate/dense/kernel:05bert/encoder/layer_1/intermediate/dense/kernel/Assign5bert/encoder/layer_1/intermediate/dense/kernel/read:02Mbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_1/intermediate/dense/bias:03bert/encoder/layer_1/intermediate/dense/bias/Assign3bert/encoder/layer_1/intermediate/dense/bias/read:02@bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_1/output/dense/kernel:0/bert/encoder/layer_1/output/dense/kernel/Assign/bert/encoder/layer_1/output/dense/kernel/read:02Gbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_1/output/dense/bias:0-bert/encoder/layer_1/output/dense/bias/Assign-bert/encoder/layer_1/output/dense/bias/read:02:bert/encoder/layer_1/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign1bert/encoder/layer_1/output/LayerNorm/beta/read:02>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/Assign2bert/encoder/layer_1/output/LayerNorm/gamma/read:02>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_2/attention/self/query/kernel:07bert/encoder/layer_2/attention/self/query/kernel/Assign7bert/encoder/layer_2/attention/self/query/kernel/read:02Obert/encoder/layer_2/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_2/attention/self/query/bias:05bert/encoder/layer_2/attention/self/query/bias/Assign5bert/encoder/layer_2/attention/self/query/bias/read:02Bbert/encoder/layer_2/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_2/attention/self/key/kernel:05bert/encoder/layer_2/attention/self/key/kernel/Assign5bert/encoder/layer_2/attention/self/key/kernel/read:02Mbert/encoder/layer_2/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_2/attention/self/key/bias:03bert/encoder/layer_2/attention/self/key/bias/Assign3bert/encoder/layer_2/attention/self/key/bias/read:02@bert/encoder/layer_2/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_2/attention/self/value/kernel:07bert/encoder/layer_2/attention/self/value/kernel/Assign7bert/encoder/layer_2/attention/self/value/kernel/read:02Obert/encoder/layer_2/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_2/attention/self/value/bias:05bert/encoder/layer_2/attention/self/value/bias/Assign5bert/encoder/layer_2/attention/self/value/bias/read:02Bbert/encoder/layer_2/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_2/attention/output/dense/kernel:09bert/encoder/layer_2/attention/output/dense/kernel/Assign9bert/encoder/layer_2/attention/output/dense/kernel/read:02Qbert/encoder/layer_2/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_2/attention/output/dense/bias:07bert/encoder/layer_2/attention/output/dense/bias/Assign7bert/encoder/layer_2/attention/output/dense/bias/read:02Dbert/encoder/layer_2/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_2/attention/output/LayerNorm/beta:0;bert/encoder/layer_2/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_2/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_2/attention/output/LayerNorm/gamma:0<bert/encoder/layer_2/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_2/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_2/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_2/intermediate/dense/kernel:05bert/encoder/layer_2/intermediate/dense/kernel/Assign5bert/encoder/layer_2/intermediate/dense/kernel/read:02Mbert/encoder/layer_2/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_2/intermediate/dense/bias:03bert/encoder/layer_2/intermediate/dense/bias/Assign3bert/encoder/layer_2/intermediate/dense/bias/read:02@bert/encoder/layer_2/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_2/output/dense/kernel:0/bert/encoder/layer_2/output/dense/kernel/Assign/bert/encoder/layer_2/output/dense/kernel/read:02Gbert/encoder/layer_2/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_2/output/dense/bias:0-bert/encoder/layer_2/output/dense/bias/Assign-bert/encoder/layer_2/output/dense/bias/read:02:bert/encoder/layer_2/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_2/output/LayerNorm/beta:01bert/encoder/layer_2/output/LayerNorm/beta/Assign1bert/encoder/layer_2/output/LayerNorm/beta/read:02>bert/encoder/layer_2/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_2/output/LayerNorm/gamma:02bert/encoder/layer_2/output/LayerNorm/gamma/Assign2bert/encoder/layer_2/output/LayerNorm/gamma/read:02>bert/encoder/layer_2/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_3/attention/self/query/kernel:07bert/encoder/layer_3/attention/self/query/kernel/Assign7bert/encoder/layer_3/attention/self/query/kernel/read:02Obert/encoder/layer_3/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_3/attention/self/query/bias:05bert/encoder/layer_3/attention/self/query/bias/Assign5bert/encoder/layer_3/attention/self/query/bias/read:02Bbert/encoder/layer_3/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_3/attention/self/key/kernel:05bert/encoder/layer_3/attention/self/key/kernel/Assign5bert/encoder/layer_3/attention/self/key/kernel/read:02Mbert/encoder/layer_3/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_3/attention/self/key/bias:03bert/encoder/layer_3/attention/self/key/bias/Assign3bert/encoder/layer_3/attention/self/key/bias/read:02@bert/encoder/layer_3/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_3/attention/self/value/kernel:07bert/encoder/layer_3/attention/self/value/kernel/Assign7bert/encoder/layer_3/attention/self/value/kernel/read:02Obert/encoder/layer_3/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_3/attention/self/value/bias:05bert/encoder/layer_3/attention/self/value/bias/Assign5bert/encoder/layer_3/attention/self/value/bias/read:02Bbert/encoder/layer_3/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_3/attention/output/dense/kernel:09bert/encoder/layer_3/attention/output/dense/kernel/Assign9bert/encoder/layer_3/attention/output/dense/kernel/read:02Qbert/encoder/layer_3/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_3/attention/output/dense/bias:07bert/encoder/layer_3/attention/output/dense/bias/Assign7bert/encoder/layer_3/attention/output/dense/bias/read:02Dbert/encoder/layer_3/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_3/attention/output/LayerNorm/beta:0;bert/encoder/layer_3/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_3/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_3/attention/output/LayerNorm/gamma:0<bert/encoder/layer_3/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_3/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_3/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_3/intermediate/dense/kernel:05bert/encoder/layer_3/intermediate/dense/kernel/Assign5bert/encoder/layer_3/intermediate/dense/kernel/read:02Mbert/encoder/layer_3/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_3/intermediate/dense/bias:03bert/encoder/layer_3/intermediate/dense/bias/Assign3bert/encoder/layer_3/intermediate/dense/bias/read:02@bert/encoder/layer_3/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_3/output/dense/kernel:0/bert/encoder/layer_3/output/dense/kernel/Assign/bert/encoder/layer_3/output/dense/kernel/read:02Gbert/encoder/layer_3/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_3/output/dense/bias:0-bert/encoder/layer_3/output/dense/bias/Assign-bert/encoder/layer_3/output/dense/bias/read:02:bert/encoder/layer_3/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_3/output/LayerNorm/beta:01bert/encoder/layer_3/output/LayerNorm/beta/Assign1bert/encoder/layer_3/output/LayerNorm/beta/read:02>bert/encoder/layer_3/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_3/output/LayerNorm/gamma:02bert/encoder/layer_3/output/LayerNorm/gamma/Assign2bert/encoder/layer_3/output/LayerNorm/gamma/read:02>bert/encoder/layer_3/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_4/attention/self/query/kernel:07bert/encoder/layer_4/attention/self/query/kernel/Assign7bert/encoder/layer_4/attention/self/query/kernel/read:02Obert/encoder/layer_4/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_4/attention/self/query/bias:05bert/encoder/layer_4/attention/self/query/bias/Assign5bert/encoder/layer_4/attention/self/query/bias/read:02Bbert/encoder/layer_4/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_4/attention/self/key/kernel:05bert/encoder/layer_4/attention/self/key/kernel/Assign5bert/encoder/layer_4/attention/self/key/kernel/read:02Mbert/encoder/layer_4/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_4/attention/self/key/bias:03bert/encoder/layer_4/attention/self/key/bias/Assign3bert/encoder/layer_4/attention/self/key/bias/read:02@bert/encoder/layer_4/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_4/attention/self/value/kernel:07bert/encoder/layer_4/attention/self/value/kernel/Assign7bert/encoder/layer_4/attention/self/value/kernel/read:02Obert/encoder/layer_4/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_4/attention/self/value/bias:05bert/encoder/layer_4/attention/self/value/bias/Assign5bert/encoder/layer_4/attention/self/value/bias/read:02Bbert/encoder/layer_4/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_4/attention/output/dense/kernel:09bert/encoder/layer_4/attention/output/dense/kernel/Assign9bert/encoder/layer_4/attention/output/dense/kernel/read:02Qbert/encoder/layer_4/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_4/attention/output/dense/bias:07bert/encoder/layer_4/attention/output/dense/bias/Assign7bert/encoder/layer_4/attention/output/dense/bias/read:02Dbert/encoder/layer_4/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_4/attention/output/LayerNorm/beta:0;bert/encoder/layer_4/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_4/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_4/attention/output/LayerNorm/gamma:0<bert/encoder/layer_4/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_4/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_4/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_4/intermediate/dense/kernel:05bert/encoder/layer_4/intermediate/dense/kernel/Assign5bert/encoder/layer_4/intermediate/dense/kernel/read:02Mbert/encoder/layer_4/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_4/intermediate/dense/bias:03bert/encoder/layer_4/intermediate/dense/bias/Assign3bert/encoder/layer_4/intermediate/dense/bias/read:02@bert/encoder/layer_4/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_4/output/dense/kernel:0/bert/encoder/layer_4/output/dense/kernel/Assign/bert/encoder/layer_4/output/dense/kernel/read:02Gbert/encoder/layer_4/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_4/output/dense/bias:0-bert/encoder/layer_4/output/dense/bias/Assign-bert/encoder/layer_4/output/dense/bias/read:02:bert/encoder/layer_4/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_4/output/LayerNorm/beta:01bert/encoder/layer_4/output/LayerNorm/beta/Assign1bert/encoder/layer_4/output/LayerNorm/beta/read:02>bert/encoder/layer_4/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_4/output/LayerNorm/gamma:02bert/encoder/layer_4/output/LayerNorm/gamma/Assign2bert/encoder/layer_4/output/LayerNorm/gamma/read:02>bert/encoder/layer_4/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_5/attention/self/query/kernel:07bert/encoder/layer_5/attention/self/query/kernel/Assign7bert/encoder/layer_5/attention/self/query/kernel/read:02Obert/encoder/layer_5/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_5/attention/self/query/bias:05bert/encoder/layer_5/attention/self/query/bias/Assign5bert/encoder/layer_5/attention/self/query/bias/read:02Bbert/encoder/layer_5/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_5/attention/self/key/kernel:05bert/encoder/layer_5/attention/self/key/kernel/Assign5bert/encoder/layer_5/attention/self/key/kernel/read:02Mbert/encoder/layer_5/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_5/attention/self/key/bias:03bert/encoder/layer_5/attention/self/key/bias/Assign3bert/encoder/layer_5/attention/self/key/bias/read:02@bert/encoder/layer_5/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_5/attention/self/value/kernel:07bert/encoder/layer_5/attention/self/value/kernel/Assign7bert/encoder/layer_5/attention/self/value/kernel/read:02Obert/encoder/layer_5/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_5/attention/self/value/bias:05bert/encoder/layer_5/attention/self/value/bias/Assign5bert/encoder/layer_5/attention/self/value/bias/read:02Bbert/encoder/layer_5/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_5/attention/output/dense/kernel:09bert/encoder/layer_5/attention/output/dense/kernel/Assign9bert/encoder/layer_5/attention/output/dense/kernel/read:02Qbert/encoder/layer_5/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_5/attention/output/dense/bias:07bert/encoder/layer_5/attention/output/dense/bias/Assign7bert/encoder/layer_5/attention/output/dense/bias/read:02Dbert/encoder/layer_5/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_5/attention/output/LayerNorm/beta:0;bert/encoder/layer_5/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_5/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_5/attention/output/LayerNorm/gamma:0<bert/encoder/layer_5/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_5/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_5/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_5/intermediate/dense/kernel:05bert/encoder/layer_5/intermediate/dense/kernel/Assign5bert/encoder/layer_5/intermediate/dense/kernel/read:02Mbert/encoder/layer_5/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_5/intermediate/dense/bias:03bert/encoder/layer_5/intermediate/dense/bias/Assign3bert/encoder/layer_5/intermediate/dense/bias/read:02@bert/encoder/layer_5/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_5/output/dense/kernel:0/bert/encoder/layer_5/output/dense/kernel/Assign/bert/encoder/layer_5/output/dense/kernel/read:02Gbert/encoder/layer_5/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_5/output/dense/bias:0-bert/encoder/layer_5/output/dense/bias/Assign-bert/encoder/layer_5/output/dense/bias/read:02:bert/encoder/layer_5/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_5/output/LayerNorm/beta:01bert/encoder/layer_5/output/LayerNorm/beta/Assign1bert/encoder/layer_5/output/LayerNorm/beta/read:02>bert/encoder/layer_5/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_5/output/LayerNorm/gamma:02bert/encoder/layer_5/output/LayerNorm/gamma/Assign2bert/encoder/layer_5/output/LayerNorm/gamma/read:02>bert/encoder/layer_5/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_6/attention/self/query/kernel:07bert/encoder/layer_6/attention/self/query/kernel/Assign7bert/encoder/layer_6/attention/self/query/kernel/read:02Obert/encoder/layer_6/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_6/attention/self/query/bias:05bert/encoder/layer_6/attention/self/query/bias/Assign5bert/encoder/layer_6/attention/self/query/bias/read:02Bbert/encoder/layer_6/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_6/attention/self/key/kernel:05bert/encoder/layer_6/attention/self/key/kernel/Assign5bert/encoder/layer_6/attention/self/key/kernel/read:02Mbert/encoder/layer_6/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_6/attention/self/key/bias:03bert/encoder/layer_6/attention/self/key/bias/Assign3bert/encoder/layer_6/attention/self/key/bias/read:02@bert/encoder/layer_6/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_6/attention/self/value/kernel:07bert/encoder/layer_6/attention/self/value/kernel/Assign7bert/encoder/layer_6/attention/self/value/kernel/read:02Obert/encoder/layer_6/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_6/attention/self/value/bias:05bert/encoder/layer_6/attention/self/value/bias/Assign5bert/encoder/layer_6/attention/self/value/bias/read:02Bbert/encoder/layer_6/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_6/attention/output/dense/kernel:09bert/encoder/layer_6/attention/output/dense/kernel/Assign9bert/encoder/layer_6/attention/output/dense/kernel/read:02Qbert/encoder/layer_6/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_6/attention/output/dense/bias:07bert/encoder/layer_6/attention/output/dense/bias/Assign7bert/encoder/layer_6/attention/output/dense/bias/read:02Dbert/encoder/layer_6/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_6/attention/output/LayerNorm/beta:0;bert/encoder/layer_6/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_6/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_6/attention/output/LayerNorm/gamma:0<bert/encoder/layer_6/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_6/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_6/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_6/intermediate/dense/kernel:05bert/encoder/layer_6/intermediate/dense/kernel/Assign5bert/encoder/layer_6/intermediate/dense/kernel/read:02Mbert/encoder/layer_6/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_6/intermediate/dense/bias:03bert/encoder/layer_6/intermediate/dense/bias/Assign3bert/encoder/layer_6/intermediate/dense/bias/read:02@bert/encoder/layer_6/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_6/output/dense/kernel:0/bert/encoder/layer_6/output/dense/kernel/Assign/bert/encoder/layer_6/output/dense/kernel/read:02Gbert/encoder/layer_6/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_6/output/dense/bias:0-bert/encoder/layer_6/output/dense/bias/Assign-bert/encoder/layer_6/output/dense/bias/read:02:bert/encoder/layer_6/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_6/output/LayerNorm/beta:01bert/encoder/layer_6/output/LayerNorm/beta/Assign1bert/encoder/layer_6/output/LayerNorm/beta/read:02>bert/encoder/layer_6/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_6/output/LayerNorm/gamma:02bert/encoder/layer_6/output/LayerNorm/gamma/Assign2bert/encoder/layer_6/output/LayerNorm/gamma/read:02>bert/encoder/layer_6/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_7/attention/self/query/kernel:07bert/encoder/layer_7/attention/self/query/kernel/Assign7bert/encoder/layer_7/attention/self/query/kernel/read:02Obert/encoder/layer_7/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_7/attention/self/query/bias:05bert/encoder/layer_7/attention/self/query/bias/Assign5bert/encoder/layer_7/attention/self/query/bias/read:02Bbert/encoder/layer_7/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_7/attention/self/key/kernel:05bert/encoder/layer_7/attention/self/key/kernel/Assign5bert/encoder/layer_7/attention/self/key/kernel/read:02Mbert/encoder/layer_7/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_7/attention/self/key/bias:03bert/encoder/layer_7/attention/self/key/bias/Assign3bert/encoder/layer_7/attention/self/key/bias/read:02@bert/encoder/layer_7/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_7/attention/self/value/kernel:07bert/encoder/layer_7/attention/self/value/kernel/Assign7bert/encoder/layer_7/attention/self/value/kernel/read:02Obert/encoder/layer_7/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_7/attention/self/value/bias:05bert/encoder/layer_7/attention/self/value/bias/Assign5bert/encoder/layer_7/attention/self/value/bias/read:02Bbert/encoder/layer_7/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_7/attention/output/dense/kernel:09bert/encoder/layer_7/attention/output/dense/kernel/Assign9bert/encoder/layer_7/attention/output/dense/kernel/read:02Qbert/encoder/layer_7/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_7/attention/output/dense/bias:07bert/encoder/layer_7/attention/output/dense/bias/Assign7bert/encoder/layer_7/attention/output/dense/bias/read:02Dbert/encoder/layer_7/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_7/attention/output/LayerNorm/beta:0;bert/encoder/layer_7/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_7/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_7/attention/output/LayerNorm/gamma:0<bert/encoder/layer_7/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_7/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_7/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_7/intermediate/dense/kernel:05bert/encoder/layer_7/intermediate/dense/kernel/Assign5bert/encoder/layer_7/intermediate/dense/kernel/read:02Mbert/encoder/layer_7/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_7/intermediate/dense/bias:03bert/encoder/layer_7/intermediate/dense/bias/Assign3bert/encoder/layer_7/intermediate/dense/bias/read:02@bert/encoder/layer_7/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_7/output/dense/kernel:0/bert/encoder/layer_7/output/dense/kernel/Assign/bert/encoder/layer_7/output/dense/kernel/read:02Gbert/encoder/layer_7/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_7/output/dense/bias:0-bert/encoder/layer_7/output/dense/bias/Assign-bert/encoder/layer_7/output/dense/bias/read:02:bert/encoder/layer_7/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_7/output/LayerNorm/beta:01bert/encoder/layer_7/output/LayerNorm/beta/Assign1bert/encoder/layer_7/output/LayerNorm/beta/read:02>bert/encoder/layer_7/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_7/output/LayerNorm/gamma:02bert/encoder/layer_7/output/LayerNorm/gamma/Assign2bert/encoder/layer_7/output/LayerNorm/gamma/read:02>bert/encoder/layer_7/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_8/attention/self/query/kernel:07bert/encoder/layer_8/attention/self/query/kernel/Assign7bert/encoder/layer_8/attention/self/query/kernel/read:02Obert/encoder/layer_8/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_8/attention/self/query/bias:05bert/encoder/layer_8/attention/self/query/bias/Assign5bert/encoder/layer_8/attention/self/query/bias/read:02Bbert/encoder/layer_8/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_8/attention/self/key/kernel:05bert/encoder/layer_8/attention/self/key/kernel/Assign5bert/encoder/layer_8/attention/self/key/kernel/read:02Mbert/encoder/layer_8/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_8/attention/self/key/bias:03bert/encoder/layer_8/attention/self/key/bias/Assign3bert/encoder/layer_8/attention/self/key/bias/read:02@bert/encoder/layer_8/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_8/attention/self/value/kernel:07bert/encoder/layer_8/attention/self/value/kernel/Assign7bert/encoder/layer_8/attention/self/value/kernel/read:02Obert/encoder/layer_8/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_8/attention/self/value/bias:05bert/encoder/layer_8/attention/self/value/bias/Assign5bert/encoder/layer_8/attention/self/value/bias/read:02Bbert/encoder/layer_8/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_8/attention/output/dense/kernel:09bert/encoder/layer_8/attention/output/dense/kernel/Assign9bert/encoder/layer_8/attention/output/dense/kernel/read:02Qbert/encoder/layer_8/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_8/attention/output/dense/bias:07bert/encoder/layer_8/attention/output/dense/bias/Assign7bert/encoder/layer_8/attention/output/dense/bias/read:02Dbert/encoder/layer_8/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_8/attention/output/LayerNorm/beta:0;bert/encoder/layer_8/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_8/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_8/attention/output/LayerNorm/gamma:0<bert/encoder/layer_8/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_8/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_8/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_8/intermediate/dense/kernel:05bert/encoder/layer_8/intermediate/dense/kernel/Assign5bert/encoder/layer_8/intermediate/dense/kernel/read:02Mbert/encoder/layer_8/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_8/intermediate/dense/bias:03bert/encoder/layer_8/intermediate/dense/bias/Assign3bert/encoder/layer_8/intermediate/dense/bias/read:02@bert/encoder/layer_8/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_8/output/dense/kernel:0/bert/encoder/layer_8/output/dense/kernel/Assign/bert/encoder/layer_8/output/dense/kernel/read:02Gbert/encoder/layer_8/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_8/output/dense/bias:0-bert/encoder/layer_8/output/dense/bias/Assign-bert/encoder/layer_8/output/dense/bias/read:02:bert/encoder/layer_8/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_8/output/LayerNorm/beta:01bert/encoder/layer_8/output/LayerNorm/beta/Assign1bert/encoder/layer_8/output/LayerNorm/beta/read:02>bert/encoder/layer_8/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_8/output/LayerNorm/gamma:02bert/encoder/layer_8/output/LayerNorm/gamma/Assign2bert/encoder/layer_8/output/LayerNorm/gamma/read:02>bert/encoder/layer_8/output/LayerNorm/gamma/Initializer/ones:08
ù
2bert/encoder/layer_9/attention/self/query/kernel:07bert/encoder/layer_9/attention/self/query/kernel/Assign7bert/encoder/layer_9/attention/self/query/kernel/read:02Obert/encoder/layer_9/attention/self/query/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_9/attention/self/query/bias:05bert/encoder/layer_9/attention/self/query/bias/Assign5bert/encoder/layer_9/attention/self/query/bias/read:02Bbert/encoder/layer_9/attention/self/query/bias/Initializer/zeros:08
ñ
0bert/encoder/layer_9/attention/self/key/kernel:05bert/encoder/layer_9/attention/self/key/kernel/Assign5bert/encoder/layer_9/attention/self/key/kernel/read:02Mbert/encoder/layer_9/attention/self/key/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_9/attention/self/key/bias:03bert/encoder/layer_9/attention/self/key/bias/Assign3bert/encoder/layer_9/attention/self/key/bias/read:02@bert/encoder/layer_9/attention/self/key/bias/Initializer/zeros:08
ù
2bert/encoder/layer_9/attention/self/value/kernel:07bert/encoder/layer_9/attention/self/value/kernel/Assign7bert/encoder/layer_9/attention/self/value/kernel/read:02Obert/encoder/layer_9/attention/self/value/kernel/Initializer/truncated_normal:08
æ
0bert/encoder/layer_9/attention/self/value/bias:05bert/encoder/layer_9/attention/self/value/bias/Assign5bert/encoder/layer_9/attention/self/value/bias/read:02Bbert/encoder/layer_9/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_9/attention/output/dense/kernel:09bert/encoder/layer_9/attention/output/dense/kernel/Assign9bert/encoder/layer_9/attention/output/dense/kernel/read:02Qbert/encoder/layer_9/attention/output/dense/kernel/Initializer/truncated_normal:08
î
2bert/encoder/layer_9/attention/output/dense/bias:07bert/encoder/layer_9/attention/output/dense/bias/Assign7bert/encoder/layer_9/attention/output/dense/bias/read:02Dbert/encoder/layer_9/attention/output/dense/bias/Initializer/zeros:08
þ
6bert/encoder/layer_9/attention/output/LayerNorm/beta:0;bert/encoder/layer_9/attention/output/LayerNorm/beta/Assign;bert/encoder/layer_9/attention/output/LayerNorm/beta/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_9/attention/output/LayerNorm/gamma:0<bert/encoder/layer_9/attention/output/LayerNorm/gamma/Assign<bert/encoder/layer_9/attention/output/LayerNorm/gamma/read:02Hbert/encoder/layer_9/attention/output/LayerNorm/gamma/Initializer/ones:08
ñ
0bert/encoder/layer_9/intermediate/dense/kernel:05bert/encoder/layer_9/intermediate/dense/kernel/Assign5bert/encoder/layer_9/intermediate/dense/kernel/read:02Mbert/encoder/layer_9/intermediate/dense/kernel/Initializer/truncated_normal:08
Þ
.bert/encoder/layer_9/intermediate/dense/bias:03bert/encoder/layer_9/intermediate/dense/bias/Assign3bert/encoder/layer_9/intermediate/dense/bias/read:02@bert/encoder/layer_9/intermediate/dense/bias/Initializer/zeros:08
Ù
*bert/encoder/layer_9/output/dense/kernel:0/bert/encoder/layer_9/output/dense/kernel/Assign/bert/encoder/layer_9/output/dense/kernel/read:02Gbert/encoder/layer_9/output/dense/kernel/Initializer/truncated_normal:08
Æ
(bert/encoder/layer_9/output/dense/bias:0-bert/encoder/layer_9/output/dense/bias/Assign-bert/encoder/layer_9/output/dense/bias/read:02:bert/encoder/layer_9/output/dense/bias/Initializer/zeros:08
Ö
,bert/encoder/layer_9/output/LayerNorm/beta:01bert/encoder/layer_9/output/LayerNorm/beta/Assign1bert/encoder/layer_9/output/LayerNorm/beta/read:02>bert/encoder/layer_9/output/LayerNorm/beta/Initializer/zeros:08
Ù
-bert/encoder/layer_9/output/LayerNorm/gamma:02bert/encoder/layer_9/output/LayerNorm/gamma/Assign2bert/encoder/layer_9/output/LayerNorm/gamma/read:02>bert/encoder/layer_9/output/LayerNorm/gamma/Initializer/ones:08
ý
3bert/encoder/layer_10/attention/self/query/kernel:08bert/encoder/layer_10/attention/self/query/kernel/Assign8bert/encoder/layer_10/attention/self/query/kernel/read:02Pbert/encoder/layer_10/attention/self/query/kernel/Initializer/truncated_normal:08
ê
1bert/encoder/layer_10/attention/self/query/bias:06bert/encoder/layer_10/attention/self/query/bias/Assign6bert/encoder/layer_10/attention/self/query/bias/read:02Cbert/encoder/layer_10/attention/self/query/bias/Initializer/zeros:08
õ
1bert/encoder/layer_10/attention/self/key/kernel:06bert/encoder/layer_10/attention/self/key/kernel/Assign6bert/encoder/layer_10/attention/self/key/kernel/read:02Nbert/encoder/layer_10/attention/self/key/kernel/Initializer/truncated_normal:08
â
/bert/encoder/layer_10/attention/self/key/bias:04bert/encoder/layer_10/attention/self/key/bias/Assign4bert/encoder/layer_10/attention/self/key/bias/read:02Abert/encoder/layer_10/attention/self/key/bias/Initializer/zeros:08
ý
3bert/encoder/layer_10/attention/self/value/kernel:08bert/encoder/layer_10/attention/self/value/kernel/Assign8bert/encoder/layer_10/attention/self/value/kernel/read:02Pbert/encoder/layer_10/attention/self/value/kernel/Initializer/truncated_normal:08
ê
1bert/encoder/layer_10/attention/self/value/bias:06bert/encoder/layer_10/attention/self/value/bias/Assign6bert/encoder/layer_10/attention/self/value/bias/read:02Cbert/encoder/layer_10/attention/self/value/bias/Initializer/zeros:08

5bert/encoder/layer_10/attention/output/dense/kernel:0:bert/encoder/layer_10/attention/output/dense/kernel/Assign:bert/encoder/layer_10/attention/output/dense/kernel/read:02Rbert/encoder/layer_10/attention/output/dense/kernel/Initializer/truncated_normal:08
ò
3bert/encoder/layer_10/attention/output/dense/bias:08bert/encoder/layer_10/attention/output/dense/bias/Assign8bert/encoder/layer_10/attention/output/dense/bias/read:02Ebert/encoder/layer_10/attention/output/dense/bias/Initializer/zeros:08

7bert/encoder/layer_10/attention/output/LayerNorm/beta:0<bert/encoder/layer_10/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_10/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/beta/Initializer/zeros:08

8bert/encoder/layer_10/attention/output/LayerNorm/gamma:0=bert/encoder/layer_10/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_10/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_10/attention/output/LayerNorm/gamma/Initializer/ones:08
õ
1bert/encoder/layer_10/intermediate/dense/kernel:06bert/encoder/layer_10/intermediate/dense/kernel/Assign6bert/encoder/layer_10/intermediate/dense/kernel/read:02Nbert/encoder/layer_10/intermediate/dense/kernel/Initializer/truncated_normal:08
â
/bert/encoder/layer_10/intermediate/dense/bias:04bert/encoder/layer_10/intermediate/dense/bias/Assign4bert/encoder/layer_10/intermediate/dense/bias/read:02Abert/encoder/layer_10/intermediate/dense/bias/Initializer/zeros:08
Ý
+bert/encoder/layer_10/output/dense/kernel:00bert/encoder/layer_10/output/dense/kernel/Assign0bert/encoder/layer_10/output/dense/kernel/read:02Hbert/encoder/layer_10/output/dense/kernel/Initializer/truncated_normal:08
Ê
)bert/encoder/layer_10/output/dense/bias:0.bert/encoder/layer_10/output/dense/bias/Assign.bert/encoder/layer_10/output/dense/bias/read:02;bert/encoder/layer_10/output/dense/bias/Initializer/zeros:08
Ú
-bert/encoder/layer_10/output/LayerNorm/beta:02bert/encoder/layer_10/output/LayerNorm/beta/Assign2bert/encoder/layer_10/output/LayerNorm/beta/read:02?bert/encoder/layer_10/output/LayerNorm/beta/Initializer/zeros:08
Ý
.bert/encoder/layer_10/output/LayerNorm/gamma:03bert/encoder/layer_10/output/LayerNorm/gamma/Assign3bert/encoder/layer_10/output/LayerNorm/gamma/read:02?bert/encoder/layer_10/output/LayerNorm/gamma/Initializer/ones:08
ý
3bert/encoder/layer_11/attention/self/query/kernel:08bert/encoder/layer_11/attention/self/query/kernel/Assign8bert/encoder/layer_11/attention/self/query/kernel/read:02Pbert/encoder/layer_11/attention/self/query/kernel/Initializer/truncated_normal:08
ê
1bert/encoder/layer_11/attention/self/query/bias:06bert/encoder/layer_11/attention/self/query/bias/Assign6bert/encoder/layer_11/attention/self/query/bias/read:02Cbert/encoder/layer_11/attention/self/query/bias/Initializer/zeros:08
õ
1bert/encoder/layer_11/attention/self/key/kernel:06bert/encoder/layer_11/attention/self/key/kernel/Assign6bert/encoder/layer_11/attention/self/key/kernel/read:02Nbert/encoder/layer_11/attention/self/key/kernel/Initializer/truncated_normal:08
â
/bert/encoder/layer_11/attention/self/key/bias:04bert/encoder/layer_11/attention/self/key/bias/Assign4bert/encoder/layer_11/attention/self/key/bias/read:02Abert/encoder/layer_11/attention/self/key/bias/Initializer/zeros:08
ý
3bert/encoder/layer_11/attention/self/value/kernel:08bert/encoder/layer_11/attention/self/value/kernel/Assign8bert/encoder/layer_11/attention/self/value/kernel/read:02Pbert/encoder/layer_11/attention/self/value/kernel/Initializer/truncated_normal:08
ê
1bert/encoder/layer_11/attention/self/value/bias:06bert/encoder/layer_11/attention/self/value/bias/Assign6bert/encoder/layer_11/attention/self/value/bias/read:02Cbert/encoder/layer_11/attention/self/value/bias/Initializer/zeros:08

5bert/encoder/layer_11/attention/output/dense/kernel:0:bert/encoder/layer_11/attention/output/dense/kernel/Assign:bert/encoder/layer_11/attention/output/dense/kernel/read:02Rbert/encoder/layer_11/attention/output/dense/kernel/Initializer/truncated_normal:08
ò
3bert/encoder/layer_11/attention/output/dense/bias:08bert/encoder/layer_11/attention/output/dense/bias/Assign8bert/encoder/layer_11/attention/output/dense/bias/read:02Ebert/encoder/layer_11/attention/output/dense/bias/Initializer/zeros:08

7bert/encoder/layer_11/attention/output/LayerNorm/beta:0<bert/encoder/layer_11/attention/output/LayerNorm/beta/Assign<bert/encoder/layer_11/attention/output/LayerNorm/beta/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/beta/Initializer/zeros:08

8bert/encoder/layer_11/attention/output/LayerNorm/gamma:0=bert/encoder/layer_11/attention/output/LayerNorm/gamma/Assign=bert/encoder/layer_11/attention/output/LayerNorm/gamma/read:02Ibert/encoder/layer_11/attention/output/LayerNorm/gamma/Initializer/ones:08
õ
1bert/encoder/layer_11/intermediate/dense/kernel:06bert/encoder/layer_11/intermediate/dense/kernel/Assign6bert/encoder/layer_11/intermediate/dense/kernel/read:02Nbert/encoder/layer_11/intermediate/dense/kernel/Initializer/truncated_normal:08
â
/bert/encoder/layer_11/intermediate/dense/bias:04bert/encoder/layer_11/intermediate/dense/bias/Assign4bert/encoder/layer_11/intermediate/dense/bias/read:02Abert/encoder/layer_11/intermediate/dense/bias/Initializer/zeros:08
Ý
+bert/encoder/layer_11/output/dense/kernel:00bert/encoder/layer_11/output/dense/kernel/Assign0bert/encoder/layer_11/output/dense/kernel/read:02Hbert/encoder/layer_11/output/dense/kernel/Initializer/truncated_normal:08
Ê
)bert/encoder/layer_11/output/dense/bias:0.bert/encoder/layer_11/output/dense/bias/Assign.bert/encoder/layer_11/output/dense/bias/read:02;bert/encoder/layer_11/output/dense/bias/Initializer/zeros:08
Ú
-bert/encoder/layer_11/output/LayerNorm/beta:02bert/encoder/layer_11/output/LayerNorm/beta/Assign2bert/encoder/layer_11/output/LayerNorm/beta/read:02?bert/encoder/layer_11/output/LayerNorm/beta/Initializer/zeros:08
Ý
.bert/encoder/layer_11/output/LayerNorm/gamma:03bert/encoder/layer_11/output/LayerNorm/gamma/Assign3bert/encoder/layer_11/output/LayerNorm/gamma/read:02?bert/encoder/layer_11/output/LayerNorm/gamma/Initializer/ones:08

bert/pooler/dense/kernel:0bert/pooler/dense/kernel/Assignbert/pooler/dense/kernel/read:027bert/pooler/dense/kernel/Initializer/truncated_normal:08

bert/pooler/dense/bias:0bert/pooler/dense/bias/Assignbert/pooler/dense/bias/read:02*bert/pooler/dense/bias/Initializer/zeros:08
q
output_weights:0output_weights/Assignoutput_weights/read:02-output_weights/Initializer/truncated_normal:08
Z
output_bias:0output_bias/Assignoutput_bias/read:02output_bias/Initializer/zeros:08*
serving_defaultð
"
	label_ids
label_ids:0
+
segment_ids
segment_ids:0	
)

input_mask
input_mask:0	
'
	input_ids
input_ids:0	-
probabilities
loss/Softmax:0tensorflow/serving/predict