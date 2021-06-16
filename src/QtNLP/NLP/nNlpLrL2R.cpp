#include <qtnlp.h>

N::NLP::LrL2R:: LrL2R      ( LrProblem * Problem , double * c )
              : LrFunction (                                  )
{
  int l   = Problem->l    ;
  problem = Problem       ;
  C       = NULL          ;
  z       = NULL          ;
  D       = NULL          ;
  if (l<=0) return        ;
  z       = new double[l] ;
  D       = new double[l] ;
  C       = c             ;
}

N::NLP::LrL2R:: LrL2R      ( void )
              : LrFunction (      )
{
  C       = NULL ;
  z       = NULL ;
  D       = NULL ;
  problem = NULL ;
}

N::NLP::LrL2R::~LrL2R(void)
{
  if (NotNull(z)) delete [] z ;
  if (NotNull(D)) delete [] D ;
  z = NULL                    ;
  D = NULL                    ;
}

double N::NLP::LrL2R::func(double * w)
{
  double   f = 0                                           ;
  double * y = problem->y                                  ;
  int      l = problem->l                                  ;
  int      v = problem->n                                  ;
  //////////////////////////////////////////////////////////
  Xv ( w , z )                                             ;
  nFullLoop ( i , v )                                      {
    f += nSquare( w[i] )                                   ;
  }                                                        ;
  f /= 2.0                                                 ;
  //////////////////////////////////////////////////////////
  nFullLoop ( i , l )                                      {
    double yz = y[i] * z[i]                                ;
    if ( yz >= 0) f += C[i] * log(1 + exp( -yz         ) ) ;
             else f += C[i] * ( -yz + log( 1 + exp(yz) ) ) ;
  }                                                        ;
  //////////////////////////////////////////////////////////
  return f                                                 ;
}

void N::NLP::LrL2R::grad(double * w,double * g)
{
  int      l = problem->l                  ;
  double * y = problem->y                  ;
  int      v = problem->n                  ;
  //////////////////////////////////////////
  nFullLoop ( i , l )                      {
    z[i] = 1    / (1 + exp( -y[i]*z[i] ) ) ;
    D[i] = z[i] * (1 - z[i]              ) ;
    z[i] = C[i] * ( z[i] - 1 ) * y[i]      ;
  }                                        ;
  //////////////////////////////////////////
  XTv ( z , g )                            ;
  //////////////////////////////////////////
  nFullLoop ( i , v ) g[i] += w[i]         ;
}

void N::NLP::LrL2R::Hv(double * s,double * Hs)
{
  int      l  = problem->l      ;
  int      w  = problem->n      ;
  double * wa = new double[l]   ;
  ///////////////////////////////
  Xv ( s , wa )                 ;
  nFullLoop ( i , l )           {
    wa[i] = C[i] * D[i] * wa[i] ;
  }                             ;
  ///////////////////////////////
  XTv ( wa , Hs )               ;
  nFullLoop ( i , w )           {
    Hs[i] += s[i]               ;
  }                             ;
  ///////////////////////////////
  delete [] wa                  ;
  wa = NULL                     ;
}

int N::NLP::LrL2R::getNrVariable(void)
{
  if (IsNull(problem)) return 0 ;
  return problem->n             ;
}

void N::NLP::LrL2R::Xv(double * v,double * Xv)
{
  int       l = problem->l                  ;
  LrNode ** x = problem->x                  ;
  ///////////////////////////////////////////
  nFullLoop ( i , l )                       {
    LrNode * s = x [ i ]                    ;
    Xv[i] = 0                               ;
    while ( s->index != -1 )                {
      Xv[i] += ( v[s->index-1] * s->value ) ;
      s++                                   ;
    }                                       ;
  }                                         ;
}

void N::NLP::LrL2R::XTv(double * v,double * XTv)
{
  int       l = problem->l                        ;
  int       w = problem->n                        ;
  LrNode ** x = problem->x                        ;
  /////////////////////////////////////////////////
  nFullLoop ( i , w ) XTv[i] = 0                  ;
  /////////////////////////////////////////////////
  nFullLoop ( i , l )                             {
    LrNode * s = x[i]                             ;
    while (s->index != -1)                        {
      XTv [ s->index - 1 ] += ( v[i] * s->value ) ;
      s++                                         ;
    }                                             ;
  }                                               ;
}
