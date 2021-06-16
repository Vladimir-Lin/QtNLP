#include <qtnlp.h>

N::NLP::LrNode:: LrNode(void)
{
}

N::NLP::LrNode:: LrNode(const LrNode & node)
{
  ME = node ;
}

N::NLP::LrNode::~LrNode(void)
{
}

N::NLP::LrNode & N::NLP::LrNode::operator = (const LrNode & node)
{
  nMemberCopy ( node , index ) ;
  nMemberCopy ( node , value ) ;
  return ME                    ;
}
