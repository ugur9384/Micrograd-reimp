class Value:
  def __init__(self,data,_children=(),_op=""):
    self.data=data
    self.grad=0
    self._op=_op
    self._prev=set(_children)
    self._backward=lambda:None
    self.op=""
  def __add__(self,other):
    out=Value(self.data+other.data,(self,other),"+")
    def _backward():
      self.grad+=1.0*out.grad
      other.grad+=1.0*out.grad
    out._backward=_backward
    return out
  def __repr__(self):
    return(f"Value={self.data} gradient={self.grad}")
  def tanh(self):
    x=self.data
    t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
    out=Value(t,(self,),"tanh")

    def _backward():
      self.grad+=out.grad*(1-t**2)
    out._backward=_backward
    return out
  def __mul__(self,other):
    out=Value(self.data*other.data,(self,other),"*")
    def _backward():
      self.grad+=out.grad*other.data
      other.grad+=out.grad*self.data
    out._backward=_backward
    return out
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

