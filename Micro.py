import math
import random
class Value:
  def __init__(self,data,_children=(),_op=""):
    self.data=data
    self.grad=0
    self._op=_op
    self._prev=set(_children)
    self._backward=lambda:None
    self.op=""

  def __add__(self,other):
    other=other if isinstance(other,Value) else Value(other)
    out=Value(self.data+other.data,(self,other),"+")
    def _backward():
      self.grad+=1.0*out.grad
      other.grad+=1.0*out.grad
    out._backward=_backward
    return out
  def __radd__(self,other):
    return self+other

  def __repr__(self):
    return(f"Value={self.data} gradient={self.grad}")
  def __neg__(self):
    return self*-1
  def __sub__(self,other):
    return self+(-other)

  def exp(self):
    x=self.data
    t=math.exp(x)
    out=Value(t,(self,),"exp")
    def _backward():
      self.grad+=out.grad*t
    out._backward=_backward
    return out
  def __pow__(self,other):
    assert isinstance(other,(int,float))
    out=(Value(self.data**other,(self,),"pow"))
    def _backward():
      self.grad+=out.grad*(other*(self.data**(other-1)))
    out._backward=_backward
    return out
  def __truediv__(self,other):
    return self*(other**-1)
  def tanh(self):
    self=self*2
    t=self.exp()-1
    y=self.exp()+1
    t=t/y
    t._op="tanh"

    return t
  def relu(self):
      out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
      def _backward():
          self.grad += (out.data > 0) * out.grad
      out._backward = _backward

      return out
  def __mul__(self,other):
    other=other if isinstance(other,Value) else Value(other)
    out=Value(self.data*other.data,(self,other),"*")
    def _backward():
      self.grad+=out.grad*other.data
      other.grad+=out.grad*self.data
    out._backward=_backward
    return out
  def __rmul__(self,other):
    return(self*other)
  def backward(self):
    topo=[]
    visited=set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    self.grad=1
    for i in reversed(topo):
      i._backward()


class Module:
  def zero_grad(self):
    for p in self.parameters():
      p.grad=0
  def parameters(self):
    return[]

class Neuron(Module):
  def __init__(self,ni):
    self.w=[Value(random.uniform(-1,1)) for x in range (ni)]
    self.b=(Value(random.uniform(-1,1)))
  def parameters(self):
    return self.w+[self.b]
  def __call__(self,x):
    act=sum((x*y) for x,y in zip(x,self.w))+self.b
    out=act.tanh()
    return out
  def __repr__(self):
    return f"Neuron with {len(self.w)} parameters)"
# Things kinda get tricky here

class Layer(Module):
  def __init__(self,nin,nout):
    self.neurons=[Neuron(nin) for x in range(nout)]

  def parameters(self):
    params=[]
    for neuron in self.neurons:
      params.extend(neuron.parameters())
    return params
  def __call__(self, x):
    out = [neuron(x) for neuron in self.neurons]
    return out[0] if len(out) == 1 else out
  def __repr__(self):
    return f"Layer, total num of neurons={len(self.neurons)}]"
  
#Things got more tricky here

class MLP(Module):
  def __init__(self,nin,nout):
    size=[nin]+nout
    self.size=size
    self.layers=[Layer(size[i],size[i+1]) for i in range(len(nout)) ]
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  def __call__(self,x):

    for layer in self.layers:
      x=layer(x)
    return x
  def __repr__(self):
    return f"MLP in the shape of{self.size}"