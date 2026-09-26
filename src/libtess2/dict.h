/*
** SGI FREE SOFTWARE LICENSE B (Version 2.0, Sept. 18, 2008) 
** Copyright (C) [dates of first publication] Silicon Graphics, Inc.
** All Rights Reserved.
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
** of the Software, and to permit persons to whom the Software is furnished to do so,
** subject to the following conditions:
** 
** The above copyright notice including the dates of first publication and either this
** permission notice or a reference to http://oss.sgi.com/projects/FreeB/ shall be
** included in all copies or substantial portions of the Software. 
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
** INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
** PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL SILICON GRAPHICS, INC.
** BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
** TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
** OR OTHER DEALINGS IN THE SOFTWARE.
** 
** Except as contained in this notice, the name of Silicon Graphics, Inc. shall not
** be used in advertising or otherwise to promote the sale, use or other dealings in
** this Software without prior written authorization from Silicon Graphics, Inc.
*/
/*
** Author: Eric Veach, July 1994.
*/

#ifndef DICT_LIST_H
#define DICT_LIST_H

#include "tesselator.h"
#include "mesh.h"

typedef struct Dict Dict;
typedef struct DictNode DictNode;

Dict *dictNewDict( TESSalloc* alloc, void *frame, int (*leq)(TESStesselator *frame, ActiveRegion *key1, ActiveRegion *key2) );

void dictDeleteDict( TESSalloc* alloc, Dict *dict );

/* Search returns the node with the smallest key greater than or equal
* to the given key.  If there is no such key, returns a node whose
* key is NULL.  Similarly, Succ(Max(d)) has a NULL key, etc.
*/
DictNode *dictSearch( Dict *dict, ActiveRegion *key );
DictNode *dictInsertBefore( Dict *dict, DictNode *node, ActiveRegion *key );
void dictDelete( Dict *dict, DictNode *node );

#define dictKey(n)	((n)->key)
#define dictSucc(n)	((n)->next)
#define dictPred(n)	((n)->prev)
#define dictMin(d)	((d)->head.next)
#define dictMax(d)	((d)->head.prev)
#define dictInsert(d,k) (dictInsertBefore((d),&(d)->head,(k)))


/*** Private data structures ***/

/* The dictionary is a sorted doubly-linked list (level 0: next/prev) with a skip list index on top of it, so that
* dictSearch() is O(log n) instead of O(n). About 1 in 4 nodes has a tower with links on levels
* 1..height-1 (level L links all the nodes whose height is greater than L). All the users of the dictionary only
* see the level 0 list.
*/
#define DICT_MAX_LEVELS 16

typedef struct DictTower DictTower;

struct DictTower {
	int height;
	DictNode *next[DICT_MAX_LEVELS]; /* [0] is unused (level 0 links are in DictNode) */
	DictNode *prev[DICT_MAX_LEVELS];
};

struct DictNode {
	ActiveRegion *key;
	DictNode *next;
	DictNode *prev;
	DictTower *tower; /* NULL if the node is only on level 0 */
};

struct Dict {
	DictNode head;
	DictTower headTower;
	int maxHeight;
	unsigned int rng;
	void *frame;
	struct BucketAlloc *nodePool;
	struct BucketAlloc *towerPool;
	int (*leq)(TESStesselator *frame, ActiveRegion *key1, ActiveRegion *key2);
};

#endif
