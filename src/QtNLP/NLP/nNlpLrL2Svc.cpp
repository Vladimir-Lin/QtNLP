#include <qtnlp.h>

N::NLP::LrL2Svc:: LrL2Svc    ( LrProblem * Problem , double * c )
                : LrFunction (                                  )
{
  int l   = Problem->l    ;
  problem = Problem       ;
  C       = NULL          ;
  z       = NULL          ;
  D       = NULL          ;
  I       = NULL          ;
  if (l<=0) return        ;
  z       = new double[l] ;
  D       = new double[l] ;
  I       = new int   [l] ;
  C       = c             ;
}

N::NLP::LrL2Svc:: LrL2Svc    ( void )
                : LrFunction (      )
{
  C       = NULL ;
  z       = NULL ;
  D       = NULL ;
  I       = NULL ;
  problem = NULL ;
}

N::NLP::LrL2Svc::~LrL2Svc(void)
{
  if (NotNull(z)) delete [] z ;
  if (NotNull(D)) delete [] D ;
  if (NotNull(I)) delete [] I ;
  z = NULL                    ;
  D = NULL                    ;
  I = NULL                    ;
}

double N::NLP::LrL2Svc::func(double * w)
{
  double   f = 0                      ;
  double * y = problem->y             ;
  int      l = problem->l             ;
  int      v = problem->n             ;
  /////////////////////////////////////
  Xv ( w , z )                        ;
  nFullLoop ( i , v )                 {
    f += nSquare( w[i] )              ;
  }                                   ;
  f /= 2.0                            ;
  /////////////////////////////////////
  nFullLoop ( i , l )                 {
    double yz = y[i] * z[i]           ;
    double d  = 1 - yz                ;
    z[i] = yz                         ;
    if (d>0) f += C[i] * nSquare( d ) ;
  }                                   ;
  /////////////////////////////////////
  return f                            ;
}

void N::NLP::LrL2Svc::grad(double * w,double * g)
{
  double * y = problem->y                   ;
  int      l = problem->l                   ;
  int      v = problem->n                   ;
  ///////////////////////////////////////////
  sizeI = 0                                 ;
  nFullLoop ( i , l )                       {
    if (z[i] < 1)                           {
      z[sizeI] = C[i] * y[i] * ( z[i] - 1 ) ;
      I[sizeI] = i                          ;
      sizeI ++                              ;
    }                                       ;
  }                                         ;
  ///////////////////////////////////////////
  subXTv ( z , g )                          ;
  ///////////////////////////////////////////
  nFullLoop ( i , v )                       {
    g[i] += g[i]                            ;
    g[i] += w[i]                            ;
  }                                         ;
}

void N::NLP::LrL2Svc::Hv(double * s,double * Hs)
{
  int      v  = problem->n        ;
  double * wa = new double[sizeI] ;
  /////////////////////////////////
  subXv ( s , wa )                ;
  nFullLoop ( i , sizeI )         {
    wa[i] = C[I[i]] * wa [i]      ;
  }                               ;
  /////////////////////////////////
  subXTv ( wa , Hs )              ;
  nFullLoop ( i , v )             {
    Hs[i] += Hs[i]                ;
    Hs[i] += s [i]                ;
  }                               ;
  /////////////////////////////////
  delete [] wa                    ;
}

int N::NLP::LrL2Svc::getNrVariable(void)
{
  if (IsNull(problem)) return 0 ;
  return problem->n             ;
}

// actually, this is the same with nNlpLrL2R
void N::NLP::LrL2Svc::Xv(double * v,double * Xv)
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

void N::NLP::LrL2Svc::subXv(double * v,double * Xv)
{
  LrNode ** x = problem->x                  ;
  ///////////////////////////////////////////
  nFullLoop ( i , sizeI )                   {
    LrNode * s = x [ I[i] ]                 ;
    Xv [ i ] = 0                            ;
    while ( s->index != -1 )                {
      Xv[i] += v[ s->index - 1 ] * s->value ;
      s++                                   ;
    }                                       ;
  }                                         ;
}

void N::NLP::LrL2Svc::subXTv(double * v,double * XTv)
{
  int       w = problem->n                    ;
  LrNode ** x = problem->x                    ;
  /////////////////////////////////////////////
  nFullLoop ( i , w ) XTv[i] = 0              ;
  nFullLoop ( i , sizeI )                     {
    LrNode * s = x [ I[i] ]                   ;
    while ( s->index != -1 )                  {
      XTv [ s->index - 1 ] += v[i] * s->value ;
      s++                                     ;
    }                                         ;
  }                                           ;
}
