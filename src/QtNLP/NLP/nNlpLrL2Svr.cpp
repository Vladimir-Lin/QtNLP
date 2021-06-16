#include <qtnlp.h>

N::NLP::LrL2Svr:: LrL2Svr (LrProblem * problem,double * C,double P)
                : LrL2Svc (            problem,         C         )
{
  p = P ;
}

N::NLP::LrL2Svr::~LrL2Svr(void)
{
}

double N::NLP::LrL2Svr::func(double * w)
{
  double * y = problem->y ;
  int      l = problem->l ;
  int      v = problem->n ;
  double   f = 0          ;
  double   d              ;
  /////////////////////////
  Xv ( w , z )            ;
  nFullLoop ( i , v )     {
    f += nSquare( w[i] )  ;
  }                       ;
  f /= 2.0                ;
  /////////////////////////
  nFullLoop ( i , l )     {
    d = z[i] - y[i]       ;
    if (d < -p)           {
      double dp = d + p   ;
      dp *= dp            ;
      f  += C[i] * dp     ;
    } else
    if (d >  p)           {
      double dp = d - p   ;
      dp *= dp            ;
      f  += C[i] * dp     ;
    }                     ;
  }                       ;
  return f                ;
}

void N::NLP::LrL2Svr::grad(double * w,double * g)
{
  double * y = problem->y     ;
  int      l = problem->l     ;
  int      v = problem->n     ;
  double   d                  ;
  /////////////////////////////
  sizeI = 0                   ;
  nFullLoop ( i , l )         {
    d = z[i] - y[i]           ;
    if (d < -p)               {
      z[sizeI] = C[i] * (d+p) ;
      I[sizeI] = i            ;
      sizeI++                 ;
    } else
    if(d >  p)                {
      z[sizeI] = C[i] * (d-p) ;
      I[sizeI] = i            ;
      sizeI++                 ;
    }                         ;
  }                           ;
  subXTv ( z , g )            ;
  /////////////////////////////
  nFullLoop ( i , v )         {
    g[i] += g[i]              ;
    g[i] += w[i]              ;
  }                           ;
}
